import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import copy
import pdb
import sys
import cv2
import yaml
import pickle
import torch
import random
import time
import importlib
import faulthandler
import numpy as np
import torch.nn as nn
from collections import OrderedDict

faulthandler.enable()
import utils
from modules.sync_batchnorm import convert_model
from seq_scripts import seq_train, seq_eval, seq_feature_generation


class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.arg.train = dict(getattr(self.arg, "train", {}) or {})
        self.arg.eval = dict(getattr(self.arg, "eval", {}) or {})
        self.arg.optimizer_args = dict(getattr(self.arg, "optimizer_args", {}) or {})
        if "num_epoch" not in self.arg.optimizer_args:
            self.arg.optimizer_args["num_epoch"] = int(self.arg.num_epoch)
        if 'ctc_zero_infinity' in self.arg.train and 'ctc_zero_infinity' not in self.arg.model_args:
            self.arg.model_args['ctc_zero_infinity'] = bool(self.arg.train['ctc_zero_infinity'])
        if 'decode_mode' not in self.arg.model_args:
            self.arg.model_args['decode_mode'] = self.arg.decode_mode
        self.validate_dataset_info()
        self.save_arg()
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)
        self.device = utils.GpuDataParallel()
        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)
        self.dataset = {}
        self.data_loader = {}
        self.gloss_dict = np.load(self.arg.dataset_info['dict_path'], allow_pickle=True).item()
        self.arg.model_args['num_classes'] = len(self.gloss_dict) + 1
        self.model, self.optimizer = self.loading()

    def validate_dataset_info(self):
        def _resolve_dir(path_value, key_name):
            if os.path.isdir(path_value):
                return path_value
            candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), path_value)
            if os.path.isdir(candidate):
                return candidate
            raise FileNotFoundError(
                "{} does not exist: {}. Please set dataset_info.{} to a valid folder."
                .format(key_name, path_value, key_name)
            )

        required_keys = ["dataset_root", "dict_path"]
        if self.arg.phase in ["train", "test"]:
            required_keys.extend(["evaluation_dir", "evaluation_prefix"])
        missing = [k for k in required_keys if k not in self.arg.dataset_info]
        if missing:
            raise ValueError(
                "Missing required dataset_info keys: {}. "
                "Please update ./configs/{}.yaml".format(", ".join(missing), self.arg.dataset)
            )
        if self.arg.phase in ["train", "test"]:
            self.arg.dataset_info["evaluation_dir"] = _resolve_dir(
                self.arg.dataset_info["evaluation_dir"], "evaluation_dir"
            )
            if "evaluation_gt_dir" in self.arg.dataset_info:
                self.arg.dataset_info["evaluation_gt_dir"] = _resolve_dir(
                    self.arg.dataset_info["evaluation_gt_dir"], "evaluation_gt_dir"
                )

    def start(self):
        if self.arg.phase == 'train':
            self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            seq_model_list = []
            dev_wer = None
            fast_dev_wer = None
            test_wer = None
            eval_cfg = self.arg.eval or {}
            fast_dev_every = int(eval_cfg.get("fast_dev_every", 1))
            full_dev_every = int(eval_cfg.get("full_dev_every", self.arg.eval_interval))
            full_test_every = int(eval_cfg.get("full_test_every", full_dev_every))
            fast_dev_mode = str(eval_cfg.get("fast_dev_mode", "dev_fast"))
            report_auxiliary = bool(eval_cfg.get("report_auxiliary", False))
            for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.num_epoch):
                save_model = epoch % self.arg.save_interval == 0
                fast_eval = (fast_dev_every > 0 and epoch % fast_dev_every == 0)
                full_dev_eval = (full_dev_every > 0 and epoch % full_dev_every == 0)
                full_test_eval = (full_test_every > 0 and epoch % full_test_every == 0)
                epoch_metric_for_ckpt = None
                # train end2end model
                seq_train(self.data_loader['train'], self.model, self.optimizer,
                          self.device, epoch, self.recoder, train_cfg=self.arg.train)
                eval_mode = fast_dev_mode if fast_dev_mode in self.data_loader else "dev"
                if fast_eval and eval_mode in self.data_loader:
                    eval_start = time.time()
                    fast_dev_wer = seq_eval(
                        self.arg, self.data_loader[eval_mode], self.model, self.device,
                        eval_mode, epoch, self.arg.work_dir, self.recoder,
                        self.arg.evaluate_tool, report_auxiliary=report_auxiliary
                    )
                    self.recoder.print_log(
                        "{} WER: {:05.2f}% (eval_time={:.1f}s)".format(
                            eval_mode, fast_dev_wer, time.time() - eval_start
                        )
                    )
                    epoch_metric_for_ckpt = fast_dev_wer

                if full_dev_eval and "dev" in self.data_loader:
                    if eval_mode == "dev" and fast_eval:
                        dev_wer = fast_dev_wer
                    else:
                        eval_start = time.time()
                        dev_wer = seq_eval(
                            self.arg, self.data_loader['dev'], self.model, self.device,
                            'dev', epoch, self.arg.work_dir, self.recoder,
                            self.arg.evaluate_tool, report_auxiliary=report_auxiliary
                        )
                        self.recoder.print_log(
                            "Dev eval time: {:.1f}s".format(time.time() - eval_start)
                        )
                    self.recoder.print_log("Dev WER: {:05.2f}%".format(dev_wer))
                    epoch_metric_for_ckpt = dev_wer
                if full_test_eval and "test" in self.data_loader:
                    eval_start = time.time()
                    test_wer = seq_eval(
                        self.arg, self.data_loader['test'], self.model, self.device,
                        'test', epoch, self.arg.work_dir, self.recoder,
                        self.arg.evaluate_tool, report_auxiliary=report_auxiliary
                    )
                    self.recoder.print_log(
                        "Test WER: {:05.2f}% (eval_time={:.1f}s)".format(
                            test_wer, time.time() - eval_start
                        )
                    )
                if save_model:
                    metric_for_ckpt = epoch_metric_for_ckpt
                    if metric_for_ckpt is None:
                        metric_for_ckpt = dev_wer if dev_wer is not None else fast_dev_wer
                    if metric_for_ckpt is None:
                        metric_for_ckpt = 100.0
                    model_path = os.path.join(
                        self.arg.work_dir,
                        "dev_{:05.2f}_epoch{}_model.pt".format(metric_for_ckpt, epoch),
                    )
                    seq_model_list.append(model_path)
                    print("seq_model_list", seq_model_list)
                    self.save_model(epoch, model_path)
        elif self.arg.phase == 'test':
            if self.arg.load_weights is None and self.arg.load_checkpoints is None:
                raise ValueError('Please appoint --load-weights.')
            self.recoder.print_log('Model:   {}.'.format(self.arg.model))
            self.recoder.print_log('Weights: {}.'.format(self.arg.load_weights))
            # train_wer = seq_eval(self.arg, self.data_loader["train_eval"], self.model, self.device,
            #                      "train", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
            report_auxiliary = bool((self.arg.eval or {}).get("report_auxiliary", False))
            dev_wer = seq_eval(self.arg, self.data_loader["dev"], self.model, self.device,
                               "dev", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool,
                               report_auxiliary=report_auxiliary)
            test_wer = seq_eval(self.arg, self.data_loader["test"], self.model, self.device,
                                "test", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool,
                                report_auxiliary=report_auxiliary)
            self.recoder.print_log('Evaluation Done.\n')
        elif self.arg.phase == "features":
            for mode in ["train", "dev", "test"]:
                seq_feature_generation(
                    self.data_loader[mode + "_eval" if mode == "train" else mode],
                    self.model, self.device, mode, self.arg.work_dir, self.recoder
                )

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open(os.path.join(self.arg.work_dir, 'config.yaml'), 'w') as f:
            yaml.dump(arg_dict, f)

    def save_model(self, epoch, save_path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
            'rng_state': self.rng.save_rng_state(),
        }, save_path)

    def loading(self):
        self.device.set_device(self.arg.device)
        print("Loading model")
        model_class = import_class(self.arg.model)
        model = model_class(
            **self.arg.model_args,
            gloss_dict=self.gloss_dict,
        )
        optimizer = utils.Optimizer(model, self.arg.optimizer_args)

        if self.arg.load_weights:
            self.load_model_weights(model, self.arg.load_weights)
        elif self.arg.load_checkpoints:
            self.load_checkpoint_weights(model, optimizer)
        model = self.model_to_device(model)
        print("Loading model finished.")
        self.load_data()
        return model, optimizer

    @staticmethod
    def _torch_load_compat(path, map_location="cpu", weights_only=None):
        load_kwargs = {}
        if map_location is not None:
            load_kwargs["map_location"] = map_location
        if weights_only is not None:
            load_kwargs["weights_only"] = weights_only
        try:
            return torch.load(path, **load_kwargs)
        except TypeError:
            # Older torch versions do not support weights_only.
            load_kwargs.pop("weights_only", None)
            return torch.load(path, **load_kwargs)
        except pickle.UnpicklingError:
            if weights_only is False:
                raise
            # PyTorch >=2.6 defaults to weights_only=True; retry with full pickle.
            load_kwargs["weights_only"] = False
            try:
                return torch.load(path, **load_kwargs)
            except TypeError:
                load_kwargs.pop("weights_only", None)
                return torch.load(path, **load_kwargs)

    def model_to_device(self, model):
        model = model.to(self.device.output_device)
        if len(self.device.gpu_list) > 1:
            model.conv2d = nn.DataParallel(
                model.conv2d,
                device_ids=self.device.gpu_list,
                output_device=self.device.output_device)
        model = convert_model(model)
        model.cuda()
        return model

    def load_model_weights(self, model, weight_path):
        state_dict = self._torch_load_compat(weight_path, map_location="cpu", weights_only=False)
        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print('Successfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
        weights = self.modified_weights(state_dict['model_state_dict'], False)
        # weights = self.modified_weights(state_dict['model_state_dict'])
        model.load_state_dict(weights, strict=True)

    @staticmethod
    def modified_weights(state_dict, modified=False):
        state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
        if not modified:
            return state_dict
        modified_dict = dict()
        return modified_dict

    def load_checkpoint_weights(self, model, optimizer):
        self.load_model_weights(model, self.arg.load_checkpoints)
        state_dict = self._torch_load_compat(
            self.arg.load_checkpoints, map_location="cpu", weights_only=False
        )

        if len(torch.cuda.get_rng_state_all()) == len(state_dict['rng_state']['cuda']):
            print("Loading random seeds...")
            self.rng.set_rng_state(state_dict['rng_state'])
        if "optimizer_state_dict" in state_dict.keys():
            print("Loading optimizer parameters...")
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            optimizer.to(self.device.output_device)
        if "scheduler_state_dict" in state_dict.keys():
            print("Loading scheduler parameters...")
            optimizer.scheduler.load_state_dict(state_dict["scheduler_state_dict"])

        self.arg.optimizer_args['start_epoch'] = state_dict["epoch"] + 1
        self.recoder.print_log("Resuming from checkpoint: epoch {self.arg.optimizer_args['start_epoch']}")

    def load_data(self):
        print("Loading data")
        self.feeder = import_class(self.arg.feeder)
        dataset_list = [("train", True), ("train_eval", False), ("dev", False), ("test", False), ("dev_fast", False)]
        for idx, (mode, train_flag) in enumerate(dataset_list):
            feeder_mode = "train" if mode == "train_eval" else mode
            if not self._mode_available(feeder_mode):
                if mode in ["dev_fast"]:
                    self.recoder.print_log("Skip optional split '{}': info file not found.".format(feeder_mode))
                    continue
                raise FileNotFoundError("Required split '{}' info file not found.".format(feeder_mode))
            arg = copy.deepcopy(self.arg.feeder_args)
            arg["prefix"] = self.arg.dataset_info['dataset_root']
            arg["mode"] = feeder_mode
            arg["transform_mode"] = train_flag
            self.dataset[mode] = self.feeder(gloss_dict=self.gloss_dict, **arg)
            self.data_loader[mode] = self.build_dataloader(self.dataset[mode], mode, train_flag)
            if hasattr(self.dataset[mode], "use_cache") and self.dataset[mode].use_cache:
                self.recoder.print_log(
                    "Split '{}' cache coverage: {}/{} ({:.2%}), cache_dir={}".format(
                        feeder_mode,
                        self.dataset[mode].cache_existing,
                        self.dataset[mode].cache_total,
                        self.dataset[mode].cache_existing / max(1, self.dataset[mode].cache_total),
                        self.dataset[mode].cache_dir,
                    )
                )
        print("Loading data finished.")

    def _mode_available(self, mode):
        dict_path = os.path.abspath(self.arg.dataset_info["dict_path"])
        csl_dir = os.path.dirname(dict_path)
        candidates = [
            os.path.join(csl_dir, "{}_info.npy".format(mode)),
            os.path.join(".", "csl100", "{}_info.npy".format(mode)),
            os.path.join("..", "csl100", "{}_info.npy".format(mode)),
        ]
        for candidate in candidates:
            if os.path.isfile(os.path.abspath(candidate)):
                return True
        return False

    def build_dataloader(self, dataset, mode, train_flag):
        worker_init_fn = None
        num_workers = int(self.arg.num_worker)
        feeder_cfg = self.arg.feeder_args if isinstance(self.arg.feeder_args, dict) else {}
        pin_memory = bool(feeder_cfg.get("pin_memory", False))
        persistent_workers = bool(feeder_cfg.get("persistent_workers", False)) and num_workers > 0
        prefetch_factor = feeder_cfg.get("prefetch_factor", None)

        if num_workers > 0:
            worker_init_fn = self.worker_init_fn
        dataloader_kwargs = {
            "dataset": dataset,
            "batch_size": self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
            "shuffle": train_flag,
            "drop_last": train_flag,
            "num_workers": num_workers,  # if train_flag else 0
            "collate_fn": self.feeder.collate_fn,
            "worker_init_fn": worker_init_fn,
        }
        if pin_memory:
            dataloader_kwargs["pin_memory"] = True
        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = persistent_workers
            if prefetch_factor is not None:
                dataloader_kwargs["prefetch_factor"] = int(prefetch_factor)
        return torch.utils.data.DataLoader(**dataloader_kwargs)

    @staticmethod
    def worker_init_fn(worker_id):
        del worker_id
        try:
            cv2.setNumThreads(0)
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass


def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod


if __name__ == '__main__':
    sparser = utils.get_parser()
    p = sparser.parse_args()
    # p.config = "baseline_iter.yaml"
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)
    processor = Processor(args)
    utils.pack_code("./", args.work_dir)
    processor.start()
