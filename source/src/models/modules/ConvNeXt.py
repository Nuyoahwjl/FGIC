import timm
import torch

from .abstract import AbstractModule

class ConvNeXtModule(AbstractModule):
    def __init__(self, num_classes=5, model_name='convnextv2_base.fcmae_ft_in1k', pretrained=True):
        super(ConvNeXtModule, self).__init__(num_classes, model_name, pretrained)
        cache_dir = 'src/cache/ConvNeXt'
        self.model = timm.create_model(model_name, cache_dir=cache_dir,
                                      pretrained=pretrained, num_classes=num_classes)
    
    def forward(self, x):
        # 实现前向传播方法
        return self.model(x)