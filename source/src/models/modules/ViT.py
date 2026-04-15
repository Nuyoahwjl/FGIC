import timm
import torch
import timm
import sys
import os

from .abstract import AbstractModule

class ViTModule(AbstractModule):
    def __init__(self, num_classes=5, model_name='vit_small_patch16_224', pretrained=True):
        super(ViTModule, self).__init__(num_classes, model_name, pretrained)
        cache_dir = 'src/cache/ViT'
        self.model = timm.create_model(model_name, cache_dir=cache_dir,
                                      pretrained=pretrained, num_classes=num_classes)
    
    def forward(self, x):
        # 实现前向传播方法
        return self.model(x)


if __name__ == '__main__':
    # 测试模型
    model = ViTModule(num_classes=5)
    print("模型创建成功！")
    # 测试前向传播
    test_input = torch.randn(1, 3, 224, 224)  # 随机测试输入
    output = model(test_input)
    print(f"模型输出形状: {output.shape}")