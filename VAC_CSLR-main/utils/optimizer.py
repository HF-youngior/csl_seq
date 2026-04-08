import pdb
import torch
import numpy as np
import torch.optim as optim
import math


class Optimizer(object):
    def __init__(self, model, optim_dict):
        self.optim_dict = optim_dict
        if self.optim_dict["optimizer"] == 'SGD':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=self.optim_dict['base_lr'],
                momentum=0.9,
                nesterov=self.optim_dict['nesterov'],
                weight_decay=self.optim_dict['weight_decay']
            )
        elif self.optim_dict["optimizer"] == 'Adam':
            self.optimizer = optim.Adam(
                # [
                #     {'params': model.conv2d.parameters(), 'lr': self.optim_dict['base_lr']*alpha},
                #     {'params': model.conv1d.parameters(), 'lr': self.optim_dict['base_lr']*alpha},
                #     {'params': model.rnn.parameters()},
                #     {'params': model.classifier.parameters()},
                # ],
                # model.conv1d.fc.parameters(),
                model.parameters(),
                lr=self.optim_dict['base_lr'],
                weight_decay=self.optim_dict['weight_decay']
            )
        else:
            raise ValueError()
        self.scheduler = self.define_lr_scheduler(self.optimizer)

    def define_lr_scheduler(self, optimizer):
        if self.optim_dict["optimizer"] not in ['SGD', 'Adam']:
            raise ValueError()

        scheduler_name = str(self.optim_dict.get("scheduler", "multistep")).lower().strip()
        milestones = [int(x) for x in self.optim_dict.get("step", [])]
        gamma = float(self.optim_dict.get("gamma", 0.2))
        base_lr = float(self.optim_dict.get("base_lr", 1e-4))
        min_lr = float(self.optim_dict.get("min_lr", 1e-6))
        warmup_epochs = max(0, int(self.optim_dict.get("warmup_epochs", 0)))
        total_epochs = int(self.optim_dict.get("num_epoch", max(milestones + [40])))
        min_factor = max(0.0, min(1.0, min_lr / max(base_lr, 1e-12)))

        def lr_lambda(epoch):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)

            if scheduler_name == "cosine":
                remaining_epochs = max(1, total_epochs - warmup_epochs)
                if remaining_epochs <= 1:
                    progress = 1.0
                else:
                    progress = float(epoch - warmup_epochs) / float(remaining_epochs - 1)
                progress = min(1.0, max(0.0, progress))
                cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_factor + (1.0 - min_factor) * cosine_factor

            # default: multistep
            decay_count = sum(epoch >= m for m in milestones)
            return max(min_factor, gamma ** decay_count)

        start_epoch = max(0, int(self.optim_dict.get("start_epoch", 0)))
        last_epoch = start_epoch - 1
        if last_epoch >= 0:
            # Required by PyTorch when initializing scheduler with last_epoch >= 0.
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def to(self, device):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
