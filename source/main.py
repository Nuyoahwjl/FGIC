from src.train import train_model
from src.detect import detect_image

import argparse
import os
import yaml
import sys
from types import SimpleNamespace
import torch as tc

# 设置huggingface国内源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def main():
    parser = argparse.ArgumentParser(description='Main entry point for training or detection')

    parser.add_argument('--mode', type=str, choices=[
        'train400_teacher',
        'train5000_teacher', 
        'train400_main', 
        'train5000_main', 
        'test400',
        'test5000'
        ], required=True, help='选择运行模式')
    
    args = parser.parse_args()
    mode = args.mode
    args_path = None
    if mode == 'train400_teacher':
        args_path = './configs/webfg400_teacher_train.yaml'
    elif mode == 'train5000_teacher':
        args_path = './configs/webinat5000_teacher_train.yaml'
    elif mode == 'train400_main':   
        args_path = './configs/webfg400_model_train.yaml'
    elif mode == 'train5000_main':
        args_path = './configs/webinat5000_model_train.yaml'
    elif mode == 'test400':
        args_path = './configs/webfg400_test.yaml'
    elif mode == 'test5000':
        args_path = './configs/webinat5000_test.yaml'
    else:
        print(f"未知的模式: {mode}")
        sys.exit(1)

    with open(args_path, 'r') as f:
        config: dict = yaml.safe_load(f)
        config = SimpleNamespace(**config)

    if mode.startswith('train'):
        # 设置环境变量，供分布式训练使用
        os.environ['MASTER_ADDR'] = config.master_addr
        os.environ['MASTER_PORT'] = config.master_port
        if config.local_rank != -1: # 表示由 torch.distributed.run 启动
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            print(f"Detected torch.distributed.run. Rank: {rank}, World Size: {world_size}")
            train_model(rank, world_size, config)
        else: # 使用 torch.multiprocessing.spawn 启动
            world_size = tc.cuda.device_count()
            if world_size == 0:
                print("没有检测到GPU，DDP训练无法进行。请确保有可用的GPU。")
            if world_size > 1:
                print(f"检测到 {world_size} 个GPU，将使用DDP进行训练。")
                tc.multiprocessing.spawn(
                    train_model,
                    args=(world_size, config), # 将解析后的args对象传递给train_model
                    nprocs=world_size,
                    join=True
                )
            else:
                print("仅检测到1个GPU，使用单GPU进行训练。")
                train_model(0, world_size, config)
    elif mode.startswith('test'):
        detect_image(
            weights=config.weights,
            data=config.data,
            output_dir=config.output_dir,
            batchsize=config.batchsize,
            model_name=config.model_name
        )
    else:
        print(f"未知的模式: {mode}")
        sys.exit(1)

if __name__ == '__main__':
    main()