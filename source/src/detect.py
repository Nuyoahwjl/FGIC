import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import ttach
import csv
from PIL import Image
import glob

from .utils.utils import check_and_create_dir

# 导入模型
from .models.FGIC_model import FGICModel

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置随机种子
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

def detect_image(weights, data='src/data/data.yaml', output_dir='output/detect_result', batchsize=1, model_name='vit_small_patch16_224'):
    """
    检测图像的函数
    
    参数:
    weights: str, 模型权重文件的路径
    data: str, 数据配置文件路径
    output_dir: str, 检测结果输出目录
    batchsize: int, 批量大小
    model_name: str, 模型名称
    """
    # 1. 从YAML文件导入数据配置
    data_config = {}
    if os.path.exists(data):
        with open(data, 'r') as f:
            data_config = yaml.safe_load(f)
        print(f'成功加载数据配置文件: {data}')
    else:
        print(f'警告: 数据配置文件不存在: {data}，使用默认配置')
    
    # 2. 从配置中获取必要信息
    nc = data_config.get('nc', 5)  # 类别数量
    names = data_config.get('names', [f'{i:04d}' for i in range(nc)])  # 类别名称
    
    # 从配置文件获取待检测图像目录
    image_dir = data_config.get('test', '')
    if not image_dir:
        print('错误: 配置文件中未找到test路径，无法确定待检测图像目录')
        return
    
    # 2. 数据预处理
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    # 3. 初始化模型并加载权重
    model = FGICModel(model_name=model_name, num_classes=nc).to(device)
    
    if os.path.exists(weights):
        try:
            model.load_state_dict(torch.load(weights, map_location=device))
            print(f'成功加载模型权重: {weights}')
        except Exception as e:
            print(f'加载模型权重失败: {e}')
            return
    else:
        print(f'错误: 模型权重文件不存在: {weights}')
        return
    
    # 4. 创建输出目录
    check_and_create_dir(output_dir)
    
    # 5. 准备检测图像
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    
    if not image_paths:
        print(f'错误: 在{image_dir}目录下未找到支持的图像文件')
        return
    
    print(f'找到{len(image_paths)}张图像待检测')
    
    # 6. 创建自定义数据集和数据加载器
    class ImageFolderDataset(torch.utils.data.Dataset):
        def __init__(self, image_paths, transform=None):
            self.image_paths = image_paths
            self.transform = transform
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_path
    
    dataset = ImageFolderDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False)

    model = ttach.ClassificationTTAWrapper(
        model=model,
        transforms=ttach.Compose([
            ttach.HorizontalFlip(),
            ttach.VerticalFlip(),
            # ttach.Scale(scales=[0.9, 1.0, 1.1]),
            ttach.Multiply(factors=[0.9, 1.0, 1.1])
        ]),
        merge_mode='mean'
    ).to(device)

    # 多显卡支持
    if torch.cuda.device_count() > 1:
        print(f'使用{torch.cuda.device_count()}块GPU进行检测')
        model = nn.DataParallel(model)
    
    # 7. 执行检测
    model.eval()
    
    # 准备CSV结果文件
    csv_path = os.path.join(output_dir, 'detection_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        # 写入表头
        # csv_writer.writerow(['图像路径', '预测类别', '预测类别索引', '置信度'])
        
        print(f'开始检测...')
        
        with torch.no_grad():
            for i, (inputs, paths) in enumerate(dataloader):
                inputs = inputs.to(device)
                
                # 前向传播
                outputs = model(inputs)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # 获取预测结果
                confidences, predicted_indices = torch.max(probabilities, 1)
                
                # 保存结果到CSV
                for j in range(len(paths)):
                    img_path = paths[j]
                    image_name = os.path.basename(img_path)
                    predicted_class = names[predicted_indices[j].item()] if predicted_indices[j].item() < len(names) else f'未知类别{predicted_indices[j].item()}'
                    confidence = confidences[j].item()
                    csv_writer.writerow([image_name, predicted_class])
                    # csv_writer.writerow([img_path, predicted_class, predicted_indices[j].item(), confidence])
                
                # 打印进度
                if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
                    print(f'处理批次 [{i + 1}/{len(dataloader)}]')
    
    print(f'检测完成！结果已保存到: {csv_path}')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='FGIC模型检测')
    parser.add_argument('--weights', type=str, required=True, help='待检测模型参数路径')
    parser.add_argument('--data', type=str, default='src/data/data.yaml', help='数据配置文件路径，默认是src/data/data.yaml')
    parser.add_argument('--output_dir', type=str, default='output/detect_result', help='检测结果输出目录，默认是output/detect_result')
    parser.add_argument('--batchsize', type=int, default=1, help='批量大小，默认是1')
    parser.add_argument('--model_name', type=str, default='vit_small_patch16_224', help='模型名称，默认是vit_small_patch16_224')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    detect_image(
        weights=args.weights,
        data=args.data,
        output_dir=args.output_dir,
        batchsize=args.batchsize,
        model_name=args.model_name
    )