# 模型模块
该文件夹下的模块分为以下几类：
- 最外层封装模块
- 基础层模块
每一种模型有不同的封装方式。`../FGIC_model.py` 中定义了 `FGICModel` 类，该类会使用本文件夹中的最外层封装模块。
为保证类 `FGICModel` 能够正确地调用最外层封装模型，这些模块需要满足统一的封装，即继承基类 `AbstractModule`。

以下是各类模块的具体封装方式：

## 最外层封装模块
最外层封装模块是指可以直接调用训练的模块，它们都需要继承基类 `AbstractModule`。
```python
class AbstractModule(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=False):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
```
例如：
```python
class ViTModule(AbstractModule):
    def __init__(self, model_name, num_classes, pretrained=False):
        super().__init__(model_name, num_classes, pretrained)
        ... # 初始化 ViT 模型结构

    def forward(self, x):
        ... # 定义前向传播过程
        return x
```
这样就保证了类 `FGICModel` 能够按照相同的方法调用如 `ViTModule` 类这样的的 `Callable` 构造模块。

## 基础层模块
基础层模块是指非最外层的模块，它们都需要继承基类 `nn.Module`。
构造一个最外层封装模块，需要调用它需要使用的基础层模块。
