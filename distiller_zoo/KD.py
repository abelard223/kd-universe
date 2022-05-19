from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss
    
class DoubleDistillKL(nn.Module):
    """Distilling the knowledge within double distill"""
    def __init__(self, T1, T2, args):
        super(DoubleDistillKL, self).__init__()
        self.T1 = T1
        self.T2 = T2
        self.cluster_efficients = args.cluster_efficients
        
    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T1, dim=1)
        p_t = F.softmax(y_t/self.T1, dim=1)
        loss_ins = F.kl_div(p_s, p_t, size_average=False) * (self.T1**2) / y_s.shape[0]
        c_s = F.log_softmax(y_s*self.T2, dim=0)
        c_t = F.softmax(y_t*self.T2, dim=0)
        loss_clu = F.kl_div(c_s.transpose(0,1), c_t.transpose(0,1), size_average=False) * (self.T2**2) / y_s.shape[1]
        return loss_ins + self.cluster_efficients*loss_clu
        
    
class KDLossv2(nn.Module):
    """Guided Knowledge Distillation Loss"""

    def __init__(self, T):
        super().__init__()
        self.t = T

    def forward(self, stu_pred, tea_pred, label):
        s = F.log_softmax(stu_pred / self.t, dim=1)
        t = F.softmax(tea_pred / self.t, dim=1)
        t_argmax = torch.argmax(t, dim=1)
        mask = torch.eq(label, t_argmax).float()
        count = (mask[mask == 1]).size(0)
        mask = mask.unsqueeze(-1)
        correct_s = s.mul(mask)
        correct_t = t.mul(mask)
        correct_t[correct_t == 0.0] = 1.0

        loss = F.kl_div(correct_s, correct_t, reduction='sum') * (self.t**2) / count
        return loss    
    
class DistillKLv2(nn.Module):
    """Distill the Knowledge using new kd loss"""
    def __init__(self, T, num_class, args):
        super(DistillKLv2, self).__init__()
        self.T = T
        self.smoothing = args.smoothing
        self.num_class = num_class
    
    def forward(self, y_s, y_t, label):
        t_argmax = torch.argmax(y_t, dim=1)
        mask = torch.eq(label, t_argmax)
        norm = torch.norm(y_t,dim=1).unsqueeze(1)
        y_t[~mask] = (1-self.smoothing)*y_t[~mask] + norm[~mask]*self.smoothing*self.get_one_hot(label,self.num_class)[~mask]
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss
        
    def get_one_hot(self, target, num_class):
        with torch.no_grad():
            one_hot = torch.zeros(target.shape[0], num_class, device=target.device).cuda()
            one_hot = one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
        return one_hot
