import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.beta import Beta
import math


class GCE_loss(nn.Module):
    def __init__(self, q=0.6, num_classes=5):
        super(GCE_loss, self).__init__()
        self.q = q
        self.num_classes = num_classes

    def forward(self, pred, labels):
        """
        pred: [N, C] 模型输出logits
        labels: [N, C] 软标签(mixup后) 或 [N] 硬标签
        """
        pred = F.softmax(pred, dim=1)
        pred = tc.clamp(pred, min=1e-7, max=1.0)  # 数值稳定性
        
        if labels.dim() == 1:  # 硬标签 [N]
            labels_one_hot = F.one_hot(labels, self.num_classes).float()
        else:  # 软标签 [N, C] (mixup后)
            labels_one_hot = labels
            
        # GCE损失计算
        if self.q == 0:
            loss = -tc.sum(labels_one_hot * tc.log(pred), dim=1)  # CE loss
        else:
            loss = (1 - tc.sum(labels_one_hot * (pred ** self.q), dim=1)) / self.q
            
        return tc.mean(loss)
    def set_q(self, new_q):
        self.q = new_q

class SoftTriple_loss(nn.Module):
    def __init__(self, la, gamma, tau, margin, dim, cN, K):
        super(SoftTriple_loss, self).__init__()
        self.la = la
        self.gamma = 1./gamma
        self.tau = tau
        self.margin = margin
        self.cN = cN
        self.K = K
        self.fc = nn.Parameter(tc.Tensor(dim, cN*K))
        self.weight = tc.zeros(cN*K, cN*K, dtype=tc.bool).cuda()
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i*K+j, i*K+j+1:(i+1)*K] = 1
        nn.init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

    def forward(self, input, target):
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc*self.gamma, dim=2)
        simClass = tc.sum(prob*simStruc, dim=2)
        marginM = tc.zeros(simClass.shape).cuda()
        marginM[tc.arange(0, marginM.shape[0]), target] = self.margin
        lossClassify = F.cross_entropy(self.la*(simClass-marginM), target)
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = tc.sum(tc.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.cN*self.K*(self.K-1.))
            return lossClassify+self.tau*reg
        else:
            return lossClassify

class SimpleQscheduler(nn.Module):
    """根据训练阶段自适应调整q参数"""
    def __init__(self, total_epoches, q_start=0.7, q_end=0.3):
        super().__init__()
        self.total_epoches = total_epoches
        self.q_start = q_start
        self.q_end = q_end
        
    def get_q(self, epoch):
        # 训练初期用大q(更鲁棒)，后期用小q(更准确)
        if epoch>self.total_epoches:
            return self.q_end
        progress = epoch / self.total_epoches
        current_q = self.q_start + (self.q_end - self.q_start) * progress
        return current_q
    
        
   