from transformers import AutoModelForImageClassification
from torch.nn import Linear
from .abstract import AbstractModule

class DINOv2Module(AbstractModule):
    def __init__(self, num_classes=5, model_name: str = "dinov2_base", pretrained=True):
        super(DINOv2Module, self).__init__(num_classes, model_name, pretrained)
        if model_name == "dinov2_base":
            model_name = "facebook/dinov2-base-imagenet1k-1-layer"
        elif model_name == "dinov2_large":
            model_name = "facebook/dinov2-large-imagenet1k-1-layer"
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            cache_dir='src/cache/DINOv2'
        )
        # 替换分类头
        self.model.classifier = Linear(self.model.classifier.in_features, num_classes, bias=True)

    def forward(self, x):
        return self.model(x).logits