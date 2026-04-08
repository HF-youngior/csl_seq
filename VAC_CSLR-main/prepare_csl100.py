import os
import shutil

# 复制corpus.txt文件到VAC_CSLR-main目录
if not os.path.exists('corpus.txt'):
    if os.path.exists('../corpus.txt'):
        shutil.copy('../corpus.txt', 'corpus.txt')
        print("已复制corpus.txt文件")
    else:
        print("错误：找不到corpus.txt文件")
        exit(1)

# 创建csl100目录
os.makedirs('csl100', exist_ok=True)

# 运行CSL100预处理脚本
print("开始预处理CSL100数据集...")
os.system('python preprocess/csl_preprocess.py --dataset-root /root/autodl-tmp/SLR_dataset/color --corpus-path corpus.txt')
print("CSL100数据集预处理完成！")
