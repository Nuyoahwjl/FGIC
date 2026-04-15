# 主模型文件
import torch
import torch.nn as nn
from .modules import *

"""
FGICModel 是一个基于 Vision Transformer (ViT) 的模型，用于图像分类任务。Callable 模型的参数如下：
- model_name: 模型名称，当前仅支持 'vit_small_patch16_224'。
- num_classes: 分类任务的类别数量。
- pretrained: 是否加载预训练权重，默认为 False。
"""
StringToCallableDict: dict[str, AbstractModule] = {
    'vit_small_patch16_224': ViTModule,
    'gft': GFTModule,
    'convnextv2_base': ConvNeXtModule,
    'convnextv2_tiny': ConvNeXtModule,
    'convnextv2_large': ConvNeXtModule,
    'dinov2_base': DINOv2Module,
    'dinov2_large': DINOv2Module,
}

class FGICModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=False):
        super().__init__()
        try:
            self.model = StringToCallableDict[model_name](model_name=model_name, 
                            num_classes=num_classes, pretrained=pretrained)
        except KeyError:
            raise ValueError(f"Model name {model_name} is not supported.")

    def forward(self, x):
        return self.model(x)
