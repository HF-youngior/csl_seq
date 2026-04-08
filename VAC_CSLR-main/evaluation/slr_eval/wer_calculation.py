import os
import pdb
from .python_wer_evaluation import wer_calculation


def _resolve_groundtruth_path(evaluate_dir, evaluate_prefix, mode, groundtruth_dir=None):
    gt_file = f"{evaluate_prefix}-{mode}.stm"
    candidates = []
    if groundtruth_dir:
        candidates.append(os.path.join(groundtruth_dir, gt_file))
    candidates.append(os.path.join(evaluate_dir, gt_file))

    # Common repo layout fallback: <repo>/csl100/*.stm
    repo_root = os.path.abspath(os.path.join(evaluate_dir, os.pardir, os.pardir))
    candidates.append(os.path.join(repo_root, "csl100", gt_file))
    candidates.append(os.path.join(os.getcwd(), "csl100", gt_file))

    checked = []
    for path in candidates:
        abs_path = os.path.abspath(path)
        if abs_path in checked:
            continue
        checked.append(abs_path)
        if os.path.isfile(abs_path):
            return abs_path
    raise FileNotFoundError(
        "Groundtruth STM not found for mode='{}', prefix='{}'. Checked: {}".format(
            mode, evaluate_prefix, checked
        )
    )


def evaluate(prefix="./", mode="dev", evaluate_dir=None, evaluate_prefix=None,
             output_file=None, output_dir=None, python_evaluate=False,
             triplet=False, token_unit="auto", groundtruth_dir=None):
    '''
    TODO  change file save path
    '''
    sclite_path = "./software/sclite"
    gt_path = _resolve_groundtruth_path(
        evaluate_dir=evaluate_dir,
        evaluate_prefix=evaluate_prefix,
        mode=mode,
        groundtruth_dir=groundtruth_dir,
    )
    print(os.getcwd())
    os.system(f"bash {evaluate_dir}/preprocess.sh {prefix + output_file} {prefix}tmp.ctm {prefix}tmp2.ctm")
    os.system(f"cat {gt_path} | sort  -k1,1 > {prefix}tmp.stm")
    # tmp2.ctm: prediction result; tmp.stm: ground-truth result
    os.system(f"python {evaluate_dir}/mergectmstm.py {prefix}tmp2.ctm {prefix}tmp.stm")
    os.system(f"cp {prefix}tmp2.ctm {prefix}out.{output_file}")
    if python_evaluate:
        ret = wer_calculation(
            gt_path,
            f"{prefix}out.{output_file}",
            token_unit=token_unit,
        )
        if triplet:
            wer_calculation(
                gt_path,
                f"{prefix}out.{output_file}",
                f"{prefix}out.{output_file}".replace(".ctm", "-conv.ctm"),
                token_unit=token_unit,
            )
        return ret
    if output_dir is not None:
        if not os.path.isdir(prefix + output_dir):
            os.makedirs(prefix + output_dir)
        os.system(
            f"{sclite_path}  -h {prefix}out.{output_file} ctm"
            f" -r {prefix}tmp.stm stm -f 0 -o sgml sum rsum pra -O {prefix + output_dir}"
        )
    else:
        os.system(
            f"{sclite_path}  -h {prefix}out.{output_file} ctm"
            f" -r {prefix}tmp.stm stm -f 0 -o sgml sum rsum pra"
        )
    ret = os.popen(
        f"{sclite_path}  -h {prefix}out.{output_file} ctm "
        f"-r {prefix}tmp.stm stm -f 0 -o dtl stdout |grep Error"
    ).readlines()[0]
    return float(ret.split("=")[1].split("%")[0])


if __name__ == "__main__":
    evaluate("output-hypothesis-dev.ctm", mode="dev")
    evaluate("output-hypothesis-test.ctm", mode="test")
