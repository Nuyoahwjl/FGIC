import os
import torch as tc
import torch.nn as nn
import torch.optim as opt
import numpy as np
import argparse
from torchvision.datasets.folder import default_loader
import yaml
from torchvision import transforms
from .utils.dataset import MyDataset
from torch.utils.data import DataLoader, DistributedSampler, Subset
from .models.FGIC_model import FGICModel
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
from cleanlab import Datalab
from torch.utils.data.dataloader import default_collate


def custom_collate(batch):
    """过滤掉损坏的数据样本"""
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return default_collate(batch)

class CleanLabTrainer:
    """
    Cleanlab 训练器，用于检测和处理标签质量问题
    """
    def __init__(self, model_config, device, num_classes, pretrained_model_path=None):  
        self.model_config = model_config  
        self.device = device
        self.num_classes = num_classes
        self.pretrained_model_path = pretrained_model_path
        
    def load_pretrained_model(self, model_path=None):
        """加载预训练模型"""
        model = FGICModel(**self.model_config)
        model.to(self.device)
        
        if model_path and os.path.exists(model_path):
            print(f"加载预训练模型: {model_path}")
            try:
                checkpoint = tc.load(model_path, map_location=self.device)
                # 处理不同的保存格式
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print("预训练模型加载成功")
            except Exception as e:
                print(f"加载预训练模型失败: {e}")
                print("将使用随机初始化的模型")
        else:
            print("未提供预训练模型路径或文件不存在，使用随机初始化的模型")
            
        return model

    def get_predictions_from_pretrained(self, dataset, batch_size=32, num_workers=4):
        print("使用预训练模型获取预测概率...")
        
        indices_file_1 = os.path.join('output', 'exp', 'subset1_indices.npy')
        indices_1 = load_clean_indices(indices_file_1)
        dataset_1 = Subset(dataset, indices_1)
        indices_file_2 = os.path.join('output', 'exp', 'subset2_indices.npy')
        indices_2 = load_clean_indices(indices_file_2)
        dataset_2 = Subset(dataset, indices_2)

        dataloader_1 = DataLoader(dataset_1, batch_size=batch_size, 
                                shuffle=False, num_workers=num_workers, 
                                collate_fn=custom_collate)
        dataloader_2 = DataLoader(dataset_2, batch_size=batch_size, 
                                shuffle=False, num_workers=num_workers, 
                                collate_fn=custom_collate)
        
        
        # 收集所有标签和预测
        all_labels = np.zeros(len(dataset), dtype=int)
        all_preds = np.zeros((len(dataset), self.num_classes), dtype=float)
        print(f"总样本数: {len(dataset)}, 类别数: {self.num_classes}")

        # 加载预训练模型
        model = self.load_pretrained_model(model_path='output/exp/CL400_1/best_model.pt')
        model.eval()
        
        print("开始预测1...")
        print(f"数据加载器2样本数: {len(dataloader_2.dataset)}")
        with tc.no_grad():
            for batch_idx, (images, labels, indexes) in enumerate(dataloader_2):
                if images is None:
                    continue
                    
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                probs = tc.softmax(outputs, dim=1)
                
                all_preds[indexes.numpy()] = probs.cpu().numpy()
                all_labels[indexes.numpy()] = labels.cpu().numpy()
                
                if (batch_idx + 1) % 100 == 0:
                    print(f"已处理 {(batch_idx + 1) * batch_size} 个样本...")
        
        print(f"预测1完成")
        
        # 清理模型资源
        tc.cuda.empty_cache()

        model = self.load_pretrained_model(model_path='output/exp/CL400_2/best_model.pt')
        model.eval()

        print("开始预测2...")
        print(f"数据加载器1样本数: {len(dataloader_1.dataset)}")
        with tc.no_grad():
            for batch_idx, (images, labels, indexes) in enumerate(dataloader_1):
                if images is None:
                    continue
                    
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                probs = tc.softmax(outputs, dim=1)
                
                all_preds[indexes.numpy()] = probs.cpu().numpy()
                all_labels[indexes.numpy()] = labels.cpu().numpy()
                
                if (batch_idx + 1) % 100 == 0:
                    print(f"已处理 {(batch_idx + 1) * batch_size} 个样本...")
        
        print(f"预测2完成")
        
        return all_preds, all_labels

    def get_cross_val_predictions(self, dataset, cv_folds=3, batch_size=32, num_workers=4):
        """
        使用交叉验证获取预测概率 - 修复版本，避免内存泄漏
        """
        print(f"开始进行 {cv_folds} 折交叉验证以获取预测概率...")
        
        n_samples = len(dataset)
        fold_size = n_samples // cv_folds
        pred_probs = np.zeros((n_samples, self.num_classes))
        
        # 收集所有标签
        all_labels = []
        for i in range(n_samples):
            try:
                _, label = dataset[i]
                all_labels.append(label)
            except Exception as e:
                print(f"加载样本 {i} 时出错: {e}")
                all_labels.append(0)  # 默认标签
        
        all_labels = np.array(all_labels)
        
        for fold in range(cv_folds):
            print(f"训练第 {fold + 1}/{cv_folds} 折...")
            
            # 划分训练和验证集，这个划分是随机的吗？？？
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < cv_folds - 1 else n_samples
            
            val_indices = list(range(start_idx, end_idx))
            train_indices = [i for i in range(n_samples) if i not in val_indices]
            
            try:
                # 创建子集
                train_subset = Subset(dataset, train_indices)
                val_subset = Subset(dataset, val_indices)
                
                # 创建数据加载器
                train_loader = DataLoader(train_subset, batch_size=batch_size, 
                                        shuffle=True, num_workers=num_workers, 
                                        collate_fn=custom_collate)
                val_loader = DataLoader(val_subset, batch_size=batch_size, 
                                    shuffle=False, num_workers=num_workers, 
                                    collate_fn=custom_collate)
                
                # 创建并训练折模型
                fold_model = FGICModel(**self.model_config)  
                fold_model.to(self.device)
                fold_model = self._train_fold(fold_model, train_loader, epochs=10)
                
                # 获取验证集预测
                fold_preds = self._get_predictions(fold_model, val_loader)
                if len(fold_preds) > 0:
                    pred_probs[val_indices] = fold_preds
                else:
                    print(f"警告: 第 {fold + 1} 折没有获得预测结果")
                    
            except Exception as e:
                print(f"第 {fold + 1} 折交叉验证失败: {e}")
                # 为失败的折设置默认预测概率
                pred_probs[val_indices] = 1.0 / self.num_classes
                
            finally:
                # 确保资源被释放
                if 'fold_model' in locals():
                    del fold_model
                tc.cuda.empty_cache()
        
        return pred_probs, all_labels
    
    def _train_fold(self, model, train_loader, epochs=10):
        """训练单个fold的模型 - 修复版本"""
        criterion = nn.CrossEntropyLoss()
        optimizer = opt.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = opt.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                if images is None:
                    continue
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            scheduler.step()
            print(f"  Fold训练 Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")
        
        return model
    
    def _get_predictions(self, model, dataloader):
        """获取模型预测概率"""
        model.eval()
        all_preds = []
        
        with tc.no_grad():
            for images, _ in dataloader:
                if images is None:
                    continue
                images = images.to(self.device)
                outputs = model(images)
                probs = tc.softmax(outputs, dim=1)
                all_preds.append(probs.cpu().numpy())
        
        return np.concatenate(all_preds, axis=0) if all_preds else np.array([])
    
    def detect_label_issues(self, pred_probs, labels, threshold=0.1):
        """检测标签问题"""
        print("使用 cleanlab 检测标签问题...")
        pred_probs = np.clip(pred_probs, 1e-6, 1-1e-6)
        pred_probs = pred_probs / pred_probs.sum(axis=1, keepdims=True)
        
        try:
            quality_scores = get_label_quality_scores(labels, pred_probs)
        except Exception as e:
            print(f"计算质量分数出错: {e}")
            quality_scores = np.ones(len(labels))

        try:
            label_issues = find_label_issues(
                labels=labels, 
                pred_probs=pred_probs, 
                n_jobs=1, 
                return_indices_ranked_by='self_confidence'
                )
            # 获取问题样本的索引（按质量分数排序）
            # issue_indices = np.where(label_issues)[0]
            print(f"检测到 {len(label_issues)} 个潜在标签问题样本，占总样本的 {len(label_issues)/len(labels)*100:.2f}%")
        except Exception as e:
            print(f"标注问题样本出错: {e}")
            label_issues = quality_scores < max(1e-6, threshold)
            # issue_indices = np.array([], dtype=int)
        
        return label_issues, quality_scores
            

    def create_clean_dataset(self, dataset, quality_scores, label_issues, remove_ratio=0.1):
        """创建清理后的数据集 - 修复版本"""
        n_samples = len(dataset)
        
        if remove_ratio <= 0:
            print("remove_ratio <= 0, 不移除任何样本")
            return dataset, list(range(n_samples))
        
        # 计算要移除的样本数量
        n_remove = int(n_samples * remove_ratio)
        
        if n_remove > 0:
            # 修复：按质量分数排序，移除质量最差的样本
            worst_indices = np.argsort(quality_scores)[:n_remove]
            clean_indices = [i for i in range(n_samples) if i not in worst_indices]
            
            print(f"基于质量分数移除了 {n_remove} 个最差样本")
            print(f"被移除样本的平均质量分数: {np.mean(quality_scores[worst_indices]):.4f}")
            print(f"保留样本的平均质量分数: {np.mean(quality_scores[clean_indices]):.4f}")
        else:
            clean_indices = list(range(n_samples))
            print("移除比例为0，不移除任何样本")
        
        print(f"清理后的数据集大小: {len(clean_indices)} (原始: {n_samples})")
        
        return Subset(dataset, clean_indices), clean_indices

def save_clean_indices(clean_indices, save_path):
    """保存清理后的数据索引"""
    np.save(save_path, np.array(clean_indices))

def load_clean_indices(save_path):
    """加载清理后的数据索引"""
    if os.path.exists(save_path):
        return np.load(save_path).tolist()
    return None

def run_cleanlab_analysis(args, full_dataset, device, nc, save_dir, is_main_process, pretrained_model_path=None):
    """运行 CleanLab 分析 - 修复版本，增强错误处理"""
    if not is_main_process:
        return None, None
    
    print("\n=== 开始 Cleanlab 标签质量检测 ===")
    
    try:
        model_config = {
            'model_name': args.model_name, 
            'num_classes': nc, 
            'pretrained': args.pretrained
        }
        
        cleanlab_trainer = CleanLabTrainer(model_config, device, nc, pretrained_model_path)
        # 选择预测方式 - 新增逻辑
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            print("使用预训练模型进行预测...")
            pred_probs, labels = cleanlab_trainer.get_predictions_from_pretrained(
                full_dataset, batch_size=args.batch_size, num_workers=args.workers
            )
        else:
            print("未找到预训练模型，使用交叉验证方式...")
            # 获取交叉验证预测概率
            pred_probs, labels = cleanlab_trainer.get_cross_val_predictions(
                full_dataset, cv_folds=args.cleanlab_cv_folds, 
                batch_size=args.batch_size, num_workers=args.workers
            )
        
        # 检查预测概率是否有效
        if pred_probs is None or len(pred_probs) == 0:
            print("警告: 无法获得有效的预测概率，跳过 Cleanlab 分析")
            return None, None
        
        # 检测标签问题
        label_issues, quality_scores = cleanlab_trainer.detect_label_issues(
            pred_probs, labels, threshold=args.cleanlab_threshold
        )

        clean_dataset, clean_indices = cleanlab_trainer.create_clean_dataset(
            full_dataset, quality_scores, label_issues, 
            remove_ratio=args.cleanlab_remove_ratio
        )
        
        # 保存cleanlab分析结果
        cleanlab_results = {
            'pred_probs': pred_probs,
            'labels': labels,
            'label_issues': label_issues,
            'quality_scores': quality_scores,
            'clean_indices': clean_indices 
        }
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, 'cleanlab_results.npz'), **cleanlab_results)
        
        # 保存清理后的索引到文件
        clean_indices_file = os.path.join(save_dir, 'clean_indices.npy')
        save_clean_indices(clean_indices, clean_indices_file)
        
        # 生成报告
        generate_cleanlab_report(args, labels, label_issues, clean_indices, quality_scores, save_dir)
        
        print("=== Cleanlab 分析完成 ===\n")
        return clean_dataset, clean_indices
        
    except Exception as e:
        print(f"CleanLab 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    finally:
        # 确保清理资源
        if 'cleanlab_trainer' in locals():
            del cleanlab_trainer
        if 'pred_probs' in locals():
            del pred_probs
        tc.cuda.empty_cache()

def generate_cleanlab_report(args, labels, issue_indices, clean_indices, quality_scores, save_dir):
    """生成 CleanLab 分析报告"""
    n_samples = len(labels)
    n_removed = n_samples - len(clean_indices)
    
    with open(os.path.join(save_dir, 'cleanlab_report.txt'), 'w') as f:
        f.write(f"Cleanlab 标签质量分析报告\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"使用的Cleanlab模型: {args.model_name}\n")
        f.write(f"交叉验证折数: {args.cleanlab_cv_folds}\n")
        f.write(f"检测阈值: {args.cleanlab_threshold}\n")
        f.write(f"移除比例设置: {args.cleanlab_remove_ratio}\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"总样本数: {len(labels)}\n")
        f.write(f"检测到的问题样本数: {len(issue_indices)}\n")
        f.write(f"问题样本比例: {len(issue_indices)/len(labels)*100:.2f}%\n")
        f.write(f"实际移除样本数: {n_removed}\n")
        f.write(f"最终训练集大小: {len(clean_indices)}\n")
        f.write(f"数据保留率: {len(clean_indices)/n_samples*100:.2f}%\n")
        f.write(f"平均标签质量分数: {np.mean(quality_scores):.4f}\n")
        f.write(f"标签质量分数标准差: {np.std(quality_scores):.4f}\n")
        
        if n_removed > 0:
            removed_indices = list(set(range(n_samples)) - set(clean_indices))
            f.write(f"被移除样本的平均质量分数: {np.mean(quality_scores[removed_indices]):.4f}\n")
            f.write(f"保留样本的平均质量分数: {np.mean(quality_scores[clean_indices]):.4f}\n")
        
        # 保存最差样本的详细信息
        f.write(f"\n质量最差的前10个样本:\n")
        worst_10 = np.argsort(quality_scores)[:min(10, len(quality_scores))]
        for i, idx in enumerate(worst_10):
            status = "已移除" if idx not in clean_indices else "保留"
            f.write(f"  {i+1}. 样本索引: {idx}, 质量分数: {quality_scores[idx]:.4f}, 状态: {status}\n")


def main(args):
    # 设置设备
    device = tc.device(args.device if tc.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    # 加载数据集
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    data_path = data_config.get('path', data_config.get('train', ''))
    nc = data_config.get('nc', 5)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    full_dataset = MyDataset(root_dir=data_path, transform=transform, num_class=nc)
     # 检查是否使用预训练模型
    use_pretrained = args.pretrained_model_path and not args.use_cv
    if use_pretrained:
        print(f"将使用预训练模型: {args.pretrained_model_path}")
    else:
        print("将使用交叉验证方式进行训练和预测")
    # 运行CleanLab分析
    clean_dataset, clean_indices = run_cleanlab_analysis(
        args, full_dataset, device, nc, args.save_dir, True, 
        args.pretrained_model_path if use_pretrained else None
    )

    # # 初始化 CleanLabTrainer
    # model_config = {'model_name': args.model_name, 'num_classes': nc, 'pretrained': args.pretrained}
    # cleanlab_trainer = CleanLabTrainer(model_config, args.device, nc)
    # # 运行 CleanLab 分析
    # pred_probs, labels = cleanlab_trainer.get_cross_val_predictions(
    #     full_dataset, cv_folds=args.cv_folds, batch_size=args.batch_size, num_workers=args.workers
    # )
    # label_issues, quality_scores, issue_indices = cleanlab_trainer.detect_label_issues(
    #     pred_probs, labels, threshold=args.threshold
    # )
    # clean_dataset, clean_indices = cleanlab_trainer.create_clean_dataset(
    #     full_dataset, quality_scores, label_issues, remove_ratio=args.remove_ratio
    # )
    # # 保存清理后的索引
    # os.makedirs(args.save_dir, exist_ok=True)
    # save_clean_indices(clean_indices, os.path.join(args.save_dir, 'clean_indices.npy'))
    # print(f"CleanLab 分析完成，清理后的索引已保存到 {args.save_dir}/clean_indices.npy")
