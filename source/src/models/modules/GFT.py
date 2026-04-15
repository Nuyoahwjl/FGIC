from .gft_model import GFT
from .abstract import AbstractModule

class GFTModule(AbstractModule):
    def __init__(self, num_classes=5, model_name='gft', pretrained=False):
        super(GFTModule, self).__init__(num_classes, model_name, pretrained)
        cache_dir = 'src/cache/GFT'
        self.model = GFT(num_classes=num_classes)
    
    def forward(self, x):
        # 实现前向传播方法
        return self.model(x)