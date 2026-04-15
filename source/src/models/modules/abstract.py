import torch.nn as nn

class AbstractModule(nn.Module):
    def __init__(self, num_classes, model_name, pretrained=False):
        super(AbstractModule, self).__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.pretrained = pretrained