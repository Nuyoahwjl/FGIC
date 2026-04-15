import torch.nn as nn

def froze_backbone(model:nn.Module):
    for param in model.parameters():
        param.requires_grad = False

    # 打印模型结构，确认冻结
    # print("Model structure after freezing backbone:")
    # print(model)

    # 解冻分类头
    if hasattr(model.model.model, 'head'):
        # ConvNeXtV2的分类头通常是head
        for param in model.model.model.head.parameters():
            param.requires_grad = True
        print("Unfroze model head parameters.")
    elif hasattr(model.model.model, 'classifier'):
        # 有些版本使用classifier
        for param in model.model.model.classifier.parameters():
            param.requires_grad = True
        print("Unfroze model classifier parameters.")
    elif hasattr(model.model.model, 'fc'):
        # 或者其他命名
        for param in model.model.model.fc.parameters():
            param.requires_grad = True
        print("Unfroze model fc parameters.")
    
    return model
