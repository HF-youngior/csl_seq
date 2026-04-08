import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def add_base_train_args(parser: argparse.ArgumentParser, default_model_path: str, exp_name: str):
    parser.add_argument("--data_path", type=str, required=True, help="dataset root")
    parser.add_argument("--dict_path", type=str, default="./dictionary.txt")
    parser.add_argument("--corpus_path", type=str, default="./corpus.txt")
    parser.add_argument("--model_path", type=str, default=default_model_path)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--sample_size", type=int, default=128)
    parser.add_argument("--sample_duration", type=int, default=32)
    parser.add_argument("--enc_hid_dim", type=int, default=512)
    parser.add_argument("--dec_hid_dim", type=int, default=512)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--gpu_id", type=str, default="0")

    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--deterministic", type=str2bool, default=True)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument("--save_each_epoch", type=str2bool, default=True)

    parser.add_argument("--use_char_level", type=str2bool, default=True)
    parser.add_argument("--use_vac_split", type=str2bool, default=True)
    parser.add_argument("--vac_root", type=str, default="./VAC_CSLR-main")
    parser.add_argument("--train_info_path", type=str, default="")
    parser.add_argument("--val_info_path", type=str, default="")
    parser.add_argument("--test_info_path", type=str, default="")
    parser.add_argument("--results_csv", type=str, default="./results/ablation_summary.csv")
    parser.add_argument("--exp_name", type=str, default=exp_name)
