import os
import torch as tc
import torch.nn as nn
import torch.optim as opt
import yaml
import shutil
import csv
from torch.utils.data import DataLoader, DistributedSampler, Subset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from timm.data.mixup import Mixup
import argparse # 导入 argparse

# 假设这些工具函数在 .utils 目录下
from .utils.utils import check_and_create_dir
from .utils.dataset import MyDataset
from .models.FGIC_model import FGICModel
from .utils.utils import load_clean_indices
from .utils.losses import GCE_loss, SimpleQscheduler

from torch.utils.data.dataloader import default_collate
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def custom_collate(batch):
    """过滤掉损坏的数据样本"""
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return default_collate(batch)

# 修改 train_model 函数以接受 rank 和 world_size，以及一个包含所有命令行参数的 args 对象
def train_model(rank, world_size, args):
    """
    训练模型的主函数，支持DDP多GPU训练

    参数:
    rank: int, 当前进程的ID (0到world_size-1)
    world_size: int, 训练中使用的总进程数
    args: argparse.Namespace, 包含所有命令行参数的对象
    """
    ddp_enabled = world_size > 1

    if ddp_enabled:
        # 1. 初始化分布式环境
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        print(f"Rank {rank}/{world_size} initialized.")

        # 设置当前进程的设备
        device = tc.device(f'cuda:{rank}')
        tc.cuda.set_device(device)
    else:
        device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
        print("单GPU或CPU模式，未启用DDP。")

    # 设置随机种子，确保每个进程有不同的随机数，但整体可复现
    tc.manual_seed(args.seed + rank)
    tc.cuda.manual_seed_all(args.seed + rank)
    random.seed(args.seed + rank)
    
    is_main_process = (rank == 0) # 判断是否是主进程

    # 1. 创建输出目录 (只有主进程创建)
    save_dir = os.path.join(args.project, args.name)
    if is_main_process:
        check_and_create_dir(save_dir)
        # 复制配置文件到输出目录
        if os.path.exists(args.data):
            shutil.copyfile(args.data, os.path.join(save_dir, 'data.yaml'))
    if ddp_enabled:
        dist.barrier() # 确保主进程创建完目录后再进行后续操作

    # 2. 读取数据配置文件
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)

    data_path = data_config.get('path', data_config.get('train', ''))
    nc = data_config.get('nc', 5) # 从数据配置中获取类别数

    val_split = args.val_split

    full_dataset = MyDataset(root_dir=data_path, num_class=nc, mode='train', resize_scale=(args.resize_scale, 1.0))
    final_dataset = full_dataset

    if args.use_cleanlab:
        clean_indices_file = args.cleanlab
        if os.path.exists(clean_indices_file):
            clean_indices = load_clean_indices(clean_indices_file)
            print(f"Rank {rank}: 加载 CleanLab 索引，数据集大小: {len(clean_indices)}")
            final_dataset = Subset(full_dataset, clean_indices)
        else:
            print(f"Rank {rank}: 未找到 CleanLab 索引文件，使用完整数据集")
            final_dataset = full_dataset

    # TODO: 找到某种数据集划分的方式，使得标签不混乱。当然不设置valid也行，那就保持现在不动，把valid相关注释掉
    g = tc.Generator()
    g.manual_seed(args.seed) # 使用相同的种子确保所有进程划分一致
    train_size = int((1 - val_split) * len(final_dataset))
    val_size = len(final_dataset) - train_size
    train_dataset, val_dataset = tc.utils.data.random_split(final_dataset, [train_size, val_size], generator=g)

    if is_main_process:
        print(f"数据集划分完成：训练集 {len(train_dataset)} 样本，验证集 {len(val_dataset)} 样本")

    if ddp_enabled:
        train_sampler = DistributedSampler(full_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False, 
                              num_workers=args.workers, collate_fn=custom_collate,
                              sampler=train_sampler, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.workers, collate_fn=custom_collate,
                            sampler=val_sampler, persistent_workers=True)

    # 4. 初始化模型
    model = FGICModel(model_name=args.model_name, num_classes=nc, pretrained=args.pretrained).to(device)
    teacher_model = None
    if args.distill:
        teacher_model = FGICModel(model_name=args.teacher_name, num_classes=nc, pretrained=False).to(device)
        if args.teacher_weight and os.path.exists(args.teacher_weight):
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            teacher_model.load_state_dict(tc.load(args.teacher_weight, map_location=map_location))
            if is_main_process:
                print(f'加载教师模型权重: {args.teacher_weight}')
        else:
            raise FileNotFoundError(f"教师模型权重文件未找到: {args.teacher_weight}")
        teacher_model.eval()  # 设置教师模型为评估模式
        for param in teacher_model.parameters():
            param.requires_grad = False  # 冻结教师模型参数

    # 5. 加载预训练权重
    if args.weight and os.path.exists(args.weight):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        model.load_state_dict(tc.load(args.weight, map_location=map_location))
        if is_main_process:
            print(f'加载预训练模型权重: {args.weight}')
    dist.barrier()

    if ddp_enabled:
        # 6. 包装模型为 DistributedDataParallel
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # 7. 设置优化器、损失函数和学习率调度器
    criterion = GCE_loss(q=args.gce_q, num_classes=nc).to(device)
    distill_loss_fn = nn.KLDivLoss(reduction='batchmean').to(device)
    # Qscheduler = SimpleQscheduler(total_epoches=args.epoch,q_start=args.q_start,q_end=args.q_end)
    # criterion = nn.CrossEntropyLoss().to(device)
    optimizer = opt.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = opt.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)
    mixup = None
    if args.mixup:
        mixup = Mixup(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            prob=args.mixup_prob, 
            switch_prob=args.mixup_switch_prob,
            mode='batch', 
            label_smoothing=0.0, 
            num_classes=nc
        )

    # 8. 设置TensorBoard日志 (只有主进程记录)
    writer = None
    if args.logger and is_main_process:
        log_dir = os.path.join(save_dir, 'logs')
        check_and_create_dir(log_dir)
        writer = SummaryWriter(log_dir=log_dir)

    # 9. 训练循环
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch_num in range(args.epoch):
        # 更新q参数
        # current_q = Qscheduler.get_q(epoch_num)
        # train_sampler.set_epoch(epoch_num)

        if is_main_process:
            print(f'\nEpoch {epoch_num+1}/{args.epoch}')

        # 训练阶段
        model.train()
        full_dataset.mode = 'train'  # 设置数据集模式为训练
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # tqdm只在主进程显示
        train_progress = tqdm(train_loader, desc=f'Training (Rank {rank})') if is_main_process else train_loader
        for images, labels in train_progress:
            if images is None:
                continue
            images, labels = images.to(device), labels.to(device)

            if args.mixup:
                images, labels = mixup(images, labels)

            outputs = model(images)
            # criterion.set_q(current_q)
            if args.distill and teacher_model is not None:
                with tc.no_grad():
                    teacher_outputs = teacher_model(images)
                # 计算蒸馏损失
                T = args.distill_temperature
                log_probs = F.log_softmax(outputs / T, dim=1)
                soft_targets = F.softmax(teacher_outputs / T, dim=1)
                distill_loss = distill_loss_fn(log_probs, soft_targets) * (T * T)

                # 计算原始损失
                gce_loss = criterion(outputs, labels)

                # 总损失为两者加权和
                loss = args.distill_alpha * distill_loss + (1 - args.distill_alpha) * gce_loss
            else:
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            batch_total = labels.size(0)
            _, predicted = tc.max(outputs.data, 1)
            if labels.dim() > 1:
                # 对于mixup标签，取最大值对应的类别作为预测
                labels = tc.argmax(labels, dim=1)
            batch_correct = (predicted == labels).sum().item()

            train_loss += batch_loss
            train_total += batch_total
            train_correct += batch_correct
            
            if is_main_process:
                train_progress.set_postfix({'loss': f'{batch_loss/batch_total:.4f}', 'acc': f'{100*batch_correct/batch_total:.2f}%'})

        # 验证阶段
        model.eval()
        full_dataset.mode = 'valid'  # 设置数据集模式为验证
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with tc.no_grad():
            val_progress = tqdm(val_loader, desc=f'Validation (Rank {rank})') if is_main_process else val_loader
            for images, labels in val_progress:
                if images is None:
                    continue
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                batch_loss = loss.item()
                batch_total = labels.size(0)
                _, predicted = tc.max(outputs.data, 1)
                batch_correct = (predicted == labels).sum().item()

                val_loss += batch_loss
                val_total += batch_total
                val_correct += batch_correct
                
                if is_main_process:
                    val_progress.set_postfix({'loss': batch_loss/batch_total, 'acc': f'{100*batch_correct/batch_total:.2f}%'})

        if ddp_enabled:
            dist.barrier()

        train_loss_tensor = tc.tensor(train_loss).to(device)
        train_correct_tensor = tc.tensor(train_correct).to(device)
        train_total_tensor = tc.tensor(train_total).to(device)
        if ddp_enabled:
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_total_tensor, op=dist.ReduceOp.SUM)
        
        val_loss_tensor = tc.tensor(val_loss).to(device)
        val_correct_tensor = tc.tensor(val_correct).to(device)
        val_total_tensor = tc.tensor(val_total).to(device)
        if ddp_enabled:
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_total_tensor, op=dist.ReduceOp.SUM)

        train_loss_aggregated = train_loss_tensor.item() / (len(train_loader) * world_size)
        val_loss_aggregated = val_loss_tensor.item() / (len(val_loader) * world_size)
        train_acc_aggregated = 100 * train_correct_tensor.item() / train_total_tensor.item()
        val_acc_aggregated = 100 * val_correct_tensor.item() / val_total_tensor.item()
        
        scheduler.step()

        if is_main_process:
            train_losses.append(train_loss_aggregated)
            val_losses.append(val_loss_aggregated)
            train_accs.append(train_acc_aggregated)
            val_accs.append(val_acc_aggregated)

            if writer:
                writer.add_scalar('Loss/Train', train_loss_aggregated, epoch_num)
                writer.add_scalar('Loss/Val', val_loss_aggregated, epoch_num)
                writer.add_scalar('Accuracy/Train', train_acc_aggregated, epoch_num)
                writer.add_scalar('Accuracy/Val', val_acc_aggregated, epoch_num)
                writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch_num)

            if val_acc_aggregated > best_val_acc:
                best_val_acc = val_acc_aggregated
                tc.save(model.module.state_dict(), os.path.join(save_dir, 'best_model.pt'))
                print(f'\n保存最佳模型，验证准确率: {best_val_acc:.2f}%')

            tc.save(model.module.state_dict(), os.path.join(save_dir, 'last_model.pt'))

            print(f'\n训练损失: {train_loss_aggregated:.4f}, 训练准确率: {train_acc_aggregated:.2f}%')
            print(f'验证损失: {val_loss_aggregated:.4f}, 验证准确率: {val_acc_aggregated:.2f}%')
            print("当前学习率:", optimizer.param_groups[0]['lr'])
            # print(f"当前的Q参数: {current_q}")
        
        if ddp_enabled:
            dist.barrier()

    if is_main_process:
        history_csv_path = os.path.join(save_dir, 'history.csv')
        with open(history_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc(%)', 'val_acc(%)'])
            for i in range(len(train_losses)):
                csv_writer.writerow([
                    i + 1,
                    f'{train_losses[i]:.6f}',
                    f'{val_losses[i]:.6f}',
                    f'{train_accs[i]:.6f}',
                    f'{val_accs[i]:.6f}'
                ])
            csv_writer.writerow([])
            csv_writer.writerow(['best_val_acc', f'{best_val_acc:.6f}'])

        print(f'训练历史已保存到: {history_csv_path}')

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='train_loss')
        plt.plot(val_losses, label='val_loss')
        plt.title('loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='train_acc')
        plt.plot(val_accs, label='val_acc')
        plt.title('accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('accuracy(%)')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'))
        plt.close()

        if writer:
            writer.close()

        print(f'\n训练完成！最佳验证准确率: {best_val_acc:.2f}%')
        print(f'训练结果保存在: {save_dir}')

    if ddp_enabled:
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description='PyTorch DDP Training')

    # 模型参数
    parser.add_argument('--model_name', type=str, default='gft', help='模型名称 (例如: gft, resnet50, vit)')
    parser.add_argument('--pretrained', action='store_true', help='是否加载预训练权重')
    parser.add_argument('--weight', type=str, default=None, help='预训练模型权重的路径')
    
    # 数据集参数
    parser.add_argument('--data', type=str, default='src/data/data.yaml', help='数据配置文件的路径')
    parser.add_argument('--img_size', type=int, default=224, help='输入图像的大小 (例如: 224)')
    parser.add_argument('--val_split', type=float, default=0.2, help='验证集划分比例')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16, help='每个GPU的批量大小')
    parser.add_argument('--epoch', type=int, default=25, help='训练总轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='初始学习率')
    parser.add_argument('--gce_q', type=float, default=0.6, help='GCE Loss的q参数')
    parser.add_argument('--q_start', type=float, default=0.7, help='GCE Loss的q起始值')
    parser.add_argument('--q_end', type=float, default=0.3, help='GCE Loss的q结束值')
    parser.add_argument('--step_size', type=int, default=10, help='学习率调度器步长')
    parser.add_argument('--gamma', type=float, default=0.1, help='学习率调度器伽马值')
    parser.add_argument('--workers', type=int, default=4, help='数据加载器的子进程数')
    parser.add_argument('--seed', type=int, default=777, help='随机种子')

    # 输出和日志参数
    parser.add_argument('--project', type=str, default='output', help='训练信息输出的根目录')
    parser.add_argument('--name', type=str, default='exp_ddp', help='本次训练的名称，将在project目录下创建同名文件夹')
    parser.add_argument('--logger', action='store_true', help='是否使用TensorBoard记录训练过程')

    # DDP 相关参数 (通常由 torch.distributed.run 自动设置)
    parser.add_argument('--local_rank', type=int, default=-1, help='当前进程的本地排名 (由 torch.distributed.run 自动设置)')
    # master_addr 和 master_port 通常在多节点训练中需要明确设置，单节点spawn通常自动处理或使用默认
    parser.add_argument('--master_addr', type=str, default='localhost', help='主节点的IP地址')
    parser.add_argument('--master_port', type=str, default='12355', help='主节点的端口号')


    args = parser.parse_args()

    # 设置环境变量，供分布式训练使用
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    # 如果是使用 torch.distributed.run 启动，local_rank 会被设置
    # 如果是使用 spawn 启动，我们需要手动获取 world_size
    if args.local_rank != -1: # 表示由 torch.distributed.run 启动
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"Detected torch.distributed.run. Rank: {rank}, World Size: {world_size}")
        train_model(rank, world_size, args)
    else: # 使用 torch.multiprocessing.spawn 启动
        world_size = tc.cuda.device_count()
        if world_size == 0:
            print("没有检测到GPU，DDP训练无法进行。请确保有可用的GPU。")
            return
        print(f"检测到 {world_size} 个GPU，将使用DDP进行训练。")
        tc.multiprocessing.spawn(
            train_model,
            args=(world_size, args), # 将解析后的args对象传递给train_model
            nprocs=world_size,
            join=True
        )


if __name__ == '__main__':
    print("开始执行模型训练 (DDP多GPU版本，带命令行参数)...")
    main()