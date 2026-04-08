#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
连续手语识别主程序
专门用于训练和测试连续手语识别模型
"""

import os
import sys
import logging
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import argparse

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import CSL_Continuous, CSL_Continuous_Char, CSL_Continuous_Char_Info
from models.Seq2Seq import Encoder, Decoder, Seq2Seq
from train import train_seq2seq
from validation import val_seq2seq
from tools import wer
from scripts.export_seq2seq_ctm import (
    _check_key_alignment,
    _collate_batch,
    _decode_predictions,
    _load_split_info,
    _read_corpus_char_vocab,
    _resolve_eval_paths,
    _run_vac_eval,
    _write_ctm,
    VACInfoSeq2SeqDataset,
)


def resolve_info_path(vac_root, explicit_path, split_name):
    if explicit_path:
        return explicit_path
    return os.path.join(vac_root, 'csl100', f'{split_name}_info.npy')


def build_unified_eval_context(args, device):
    if not args.run_unified_eval:
        return None
    if not args.use_char_level:
        raise NotImplementedError('Unified VAC WER is currently implemented for char-level training only.')

    vac_root = os.path.abspath(args.vac_root)
    split = str(args.unified_eval_split).strip()
    split_override = {
        'dev': args.val_info_path,
        'test': args.test_info_path,
        'train': args.train_info_path,
    }.get(split, '')
    info_path = os.path.abspath(resolve_info_path(vac_root, split_override, split))
    output_dir = os.path.join(args.model_path, "unified_eval")
    os.makedirs(output_dir, exist_ok=True)

    token_to_id, id_to_token, corpus_max_len = _read_corpus_char_vocab(os.path.abspath(args.corpus_path))
    samples = _load_split_info(info_path)
    dataset = VACInfoSeq2SeqDataset(
        samples=samples,
        dataset_root=os.path.abspath(args.data_path),
        frames=int(args.sample_duration),
        sample_size=int(args.sample_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.unified_eval_batch_size),
        shuffle=False,
        num_workers=int(args.unified_eval_num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=_collate_batch,
    )
    evaluation_dir, evaluation_gt_dir = _resolve_eval_paths(
        vac_root, args.unified_evaluation_dir, args.unified_evaluation_gt_dir
    )
    return {
        "vac_root": vac_root,
        "split": split,
        "info_path": info_path,
        "output_dir": output_dir,
        "loader": loader,
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "decode_len": int(corpus_max_len),
        "device": device,
        "evaluation_dir": evaluation_dir,
        "evaluation_gt_dir": evaluation_gt_dir,
        "evaluation_prefix": args.unified_evaluation_prefix,
        "empty_token": args.unified_empty_token,
    }


def run_unified_eval(model, eval_ctx, epoch, logger, writer):
    if eval_ctx is None:
        return None

    start_time = time.time()
    split = eval_ctx["split"]
    output_ctm = os.path.join(
        eval_ctx["output_dir"],
        "epoch{:03d}_output-hypothesis-{}.ctm".format(epoch + 1, split),
    )

    logger.info("Running unified VAC WER on split='{}' ...".format(split))
    was_training = model.training
    model.eval()
    fileids, predictions = _decode_predictions(
        model=model,
        loader=eval_ctx["loader"],
        device=eval_ctx["device"],
        decode_len=eval_ctx["decode_len"],
        pad_id=eval_ctx["token_to_id"]["<pad>"],
        sos_id=eval_ctx["token_to_id"]["<sos>"],
        id_to_token=eval_ctx["id_to_token"],
    )
    _write_ctm(output_ctm, fileids, predictions, eval_ctx["empty_token"])
    _check_key_alignment(
        ctm_path=output_ctm,
        split=split,
        evaluation_prefix=eval_ctx["evaluation_prefix"],
        evaluation_gt_dir=eval_ctx["evaluation_gt_dir"],
    )
    wer_value = _run_vac_eval(
        ctm_path=output_ctm,
        split=split,
        evaluation_dir=eval_ctx["evaluation_dir"],
        evaluation_prefix=eval_ctx["evaluation_prefix"],
        evaluation_gt_dir=eval_ctx["evaluation_gt_dir"],
    )
    elapsed = time.time() - start_time
    writer.add_scalar('WER_unified/{}'.format(split), wer_value, epoch + 1)
    logger.info(
        "Unified VAC WER [{}] epoch {}: {:.2f}% (elapsed {:.1f} min)".format(
            split, epoch + 1, wer_value, elapsed / 60.0
        )
    )
    if was_training:
        model.train()
    return {
        "wer": wer_value,
        "ctm_path": output_ctm,
        "elapsed_sec": elapsed,
        "split": split,
    }

def create_corpus_file(dictionary_path, output_path):
    """
    根据字典文件创建语料库文件
    语料库格式：句子ID \t 句子内容
    """
    print(f"正在从 {dictionary_path} 创建语料库文件...")
    
    # 读取字典文件
    with open(dictionary_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 创建语料库
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, line in enumerate(lines):
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                sentence_id = f"{i:06d}"
                word = parts[1]
                # 创建简单的句子（这里可以根据需要修改）
                sentence = word  # 或者可以创建更复杂的句子
                f.write(f"{sentence_id}\t{sentence}\n")
    
    print(f"语料库文件已创建：{output_path}")

def main():
    parser = argparse.ArgumentParser(description='连续手语识别训练和测试')
    parser.add_argument('--mode', choices=['train', 'test'], default='train', 
                       help='运行模式：train（训练）或 test（测试）')
    parser.add_argument('--data_path', type=str, required=True,
                       help='连续手语数据路径')
    parser.add_argument('--dict_path', type=str, default='dictionary.txt',
                       help='字典文件路径')
    parser.add_argument('--corpus_path', type=str, default='corpus.txt',
                       help='语料库文件路径')
    parser.add_argument('--model_path', type=str, default='./models/continuous',
                       help='模型保存路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='测试时使用的模型检查点路径')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--sample_size', type=int, default=128,
                       help='图像尺寸')
    parser.add_argument('--sample_duration', type=int, default=48,
                       help='视频帧数')
    parser.add_argument('--enc_hid_dim', type=int, default=512,
                       help='编码器隐藏维度')
    parser.add_argument('--dec_hid_dim', type=int, default=512,
                       help='解码器隐藏维度')
    parser.add_argument('--emb_dim', type=int, default=256,
                       help='嵌入维度')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout率')
    parser.add_argument('--clip', type=float, default=1.0,
                       help='梯度裁剪')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='日志间隔')
    parser.add_argument('--gpu_id', type=str, default='0',
                       help='使用的GPU ID')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader worker count')
    parser.add_argument('--use_vac_split', action='store_true',
                       help='Use VAC csl100 info.npy split files')
    parser.add_argument('--vac_root', type=str, default='./VAC_CSLR-main',
                       help='VAC_CSLR-main root path')
    parser.add_argument('--train_info_path', type=str, default='',
                       help='Override train_info.npy path')
    parser.add_argument('--val_info_path', type=str, default='',
                       help='Override dev_info.npy path')
    parser.add_argument('--test_info_path', type=str, default='',
                       help='Override test_info.npy path')
    parser.add_argument('--run_unified_eval', action='store_true',
                       help='Run VAC unified WER after validation')
    parser.add_argument('--unified_eval_split', type=str, default='dev',
                       help='Split used for unified VAC WER: dev/test/train')
    parser.add_argument('--unified_eval_batch_size', type=int, default=32,
                       help='Batch size for unified VAC WER export')
    parser.add_argument('--unified_eval_num_workers', type=int, default=2,
                       help='Num workers for unified VAC WER export')
    parser.add_argument('--unified_evaluation_dir', type=str, default='./evaluation/slr_eval',
                       help='VAC evaluation directory')
    parser.add_argument('--unified_evaluation_prefix', type=str, default='csl100-groundtruth',
                       help='VAC evaluation prefix')
    parser.add_argument('--unified_evaluation_gt_dir', type=str, default='./csl100',
                       help='VAC evaluation STM directory')
    parser.add_argument('--unified_empty_token', type=str, default='<eps>',
                       help='Placeholder token for empty predictions in CTM')
    parser.add_argument('--use_char_level', action='store_true',
                       help='使用字符级别而不是词级别')
    
    args = parser.parse_args()
    
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建必要的目录
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs("log", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    
    # 创建语料库文件（如果不存在）
    if not os.path.exists(args.corpus_path):
        if os.path.exists(args.dict_path):
            create_corpus_file(args.dict_path, args.corpus_path)
        else:
            print(f"错误：找不到字典文件 {args.dict_path}")
            return
    
    if args.mode == 'train':
        train_model(args, device)
    else:
        test_model(args, device)

def train_model(args, device):
    """训练模型"""
    print("开始训练连续手语识别模型...")
    
    # 设置日志
    log_path = f"log/continuous_slr_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    sum_path = f"runs/continuous_slr_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(message)s', 
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    logger = logging.getLogger('Continuous_SLR')
    logger.info('开始训练连续手语识别模型...')
    writer = SummaryWriter(sum_path)
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize([args.sample_size, args.sample_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # ?????
    if args.use_vac_split:
        if not args.use_char_level:
            raise NotImplementedError('VAC info split is currently implemented for char-level training only.')
        vac_root = os.path.abspath(args.vac_root)
        train_info_path = os.path.abspath(resolve_info_path(vac_root, args.train_info_path, 'train'))
        val_info_path = os.path.abspath(resolve_info_path(vac_root, args.val_info_path, 'dev'))
        logger.info(f"Using VAC 8:2 metadata split: train={train_info_path}, val={val_info_path}")
        train_set = CSL_Continuous_Char_Info(
            dataset_root=args.data_path,
            corpus_path=args.corpus_path,
            info_path=train_info_path,
            frames=args.sample_duration,
            transform=transform
        )
        val_set = CSL_Continuous_Char_Info(
            dataset_root=args.data_path,
            corpus_path=args.corpus_path,
            info_path=val_info_path,
            frames=args.sample_duration,
            transform=transform
        )
    elif args.use_char_level:
        logger.info("?????????")
        train_set = CSL_Continuous_Char(
            data_path=args.data_path,
            corpus_path=args.corpus_path,
            frames=args.sample_duration,
            train=True,
            transform=transform
        )
        val_set = CSL_Continuous_Char(
            data_path=args.data_path,
            corpus_path=args.corpus_path,
            frames=args.sample_duration,
            train=False,
            transform=transform
        )
    else:
        logger.info("????????")
        train_set = CSL_Continuous(
            data_path=args.data_path,
            dict_path=args.dict_path,
            corpus_path=args.corpus_path,
            frames=args.sample_duration,
            train=True,
            transform=transform
        )
        val_set = CSL_Continuous(
            data_path=args.data_path,
            dict_path=args.dict_path,
            corpus_path=args.corpus_path,
            frames=args.sample_duration,
            train=False,
            transform=transform
        )
    
    logger.info(f"数据集样本数: 训练集={len(train_set)}, 验证集={len(val_set)}")
    logger.info(f"输出维度: {train_set.output_dim}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    unified_eval_ctx = build_unified_eval_context(args, device)

    # 创建模型
    encoder = Encoder(lstm_hidden_size=args.enc_hid_dim, arch="resnet18").to(device)
    decoder = Decoder(
        output_dim=train_set.output_dim,
        emb_dim=args.emb_dim,
        enc_hid_dim=args.enc_hid_dim,
        dec_hid_dim=args.dec_hid_dim,
        dropout=args.dropout
    ).to(device)
    model = Seq2Seq(encoder=encoder, decoder=decoder, device=device).to(device)
    
    # 多GPU支持
    if torch.cuda.device_count() > 1:
        logger.info(f"使用 {torch.cuda.device_count()} 个GPU")
        model = nn.DataParallel(model)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    # 开始训练
    logger.info("训练开始".center(60, '#'))
    best_wer = float('inf')
    
    for epoch in range(args.epochs):
        train_metrics = train_seq2seq(
            model, criterion, optimizer, args.clip,
            train_loader, device, epoch, logger, args.log_interval, writer
        )

        val_metrics = val_seq2seq(
            model, criterion, val_loader, device, epoch, logger, writer
        )

        unified_metrics = run_unified_eval(model, unified_eval_ctx, epoch, logger, writer)

        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

        model_save_path = os.path.join(args.model_path, f"continuous_slr_epoch{epoch+1:03d}.pth")
        torch.save(model_state, model_save_path)
        logger.info(f"Epoch {epoch+1} model saved: {model_save_path}")

        metric_name = 'val_WER'
        metric_value = val_metrics['wer']
        if unified_metrics is not None:
            metric_name = f"unified_{unified_metrics['split']}_WER"
            metric_value = unified_metrics['wer']

        if metric_value < best_wer:
            best_wer = metric_value
            best_model_path = os.path.join(args.model_path, "best_model.pth")
            torch.save(model_state, best_model_path)
            logger.info(f"Best model updated: {best_model_path} | {metric_name}={best_wer:.2f}%")
    
    logger.info("训练完成".center(60, '#'))
    writer.close()

def test_model(args, device):
    """测试模型"""
    print("开始测试连续手语识别模型...")
    
    if args.checkpoint is None:
        print("错误：测试模式需要指定 --checkpoint 参数")
        return
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize([args.sample_size, args.sample_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # ?????
    if args.use_vac_split:
        if not args.use_char_level:
            raise NotImplementedError('VAC info split is currently implemented for char-level training only.')
        vac_root = os.path.abspath(args.vac_root)
        test_info_path = os.path.abspath(resolve_info_path(vac_root, args.test_info_path, 'test'))
        print(f"Using VAC test metadata: {test_info_path}")
        test_set = CSL_Continuous_Char_Info(
            dataset_root=args.data_path,
            corpus_path=args.corpus_path,
            info_path=test_info_path,
            frames=args.sample_duration,
            transform=transform
        )
    elif args.use_char_level:
        test_set = CSL_Continuous_Char(
            data_path=args.data_path,
            corpus_path=args.corpus_path,
            frames=args.sample_duration,
            train=False,
            transform=transform
        )
    else:
        test_set = CSL_Continuous(
            data_path=args.data_path,
            dict_path=args.dict_path,
            corpus_path=args.corpus_path,
            frames=args.sample_duration,
            train=False,
            transform=transform
        )
    
    test_loader = DataLoader(
        test_set, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    # 创建模型
    encoder = Encoder(lstm_hidden_size=args.enc_hid_dim, arch="resnet18").to(device)
    decoder = Decoder(
        output_dim=test_set.output_dim,
        emb_dim=args.emb_dim,
        enc_hid_dim=args.enc_hid_dim,
        dec_hid_dim=args.dec_hid_dim,
        dropout=args.dropout
    ).to(device)
    model = Seq2Seq(encoder=encoder, decoder=decoder, device=device).to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    # 测试
    print("开始测试...")
    test_wer = 0.0
    test_acc = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (imgs, target) in enumerate(test_loader):
            imgs = imgs.to(device)
            target = target.to(device)
            
            # 前向传播（无教师强制）
            outputs = model(imgs, target, 0)
            
            # 计算WER和准确率
            output_dim = outputs.shape[-1]
            outputs = outputs[1:].view(-1, output_dim)
            target_flat = target.permute(1,0)[1:].reshape(-1)
            
            prediction = torch.max(outputs, 1)[1]
            
            # 计算WER
            batch_size = imgs.shape[0]
            prediction_seq = prediction.view(-1, batch_size).permute(1,0).tolist()
            target_seq = target_flat.view(-1, batch_size).permute(1,0).tolist()
            
            for i in range(batch_size):
                # 移除特殊标记
                pred_clean = [item for item in prediction_seq[i] if item not in [0,1,2]]
                target_clean = [item for item in target_seq[i] if item not in [0,1,2]]
                
                if len(target_clean) > 0:
                    test_wer += wer(target_clean, pred_clean)
                    total_samples += 1
    
    test_wer /= total_samples
    print(f"测试完成!")
    print(f"平均WER: {test_wer:.2f}%")
    print(f"测试样本数: {total_samples}")

if __name__ == '__main__':
    main()
