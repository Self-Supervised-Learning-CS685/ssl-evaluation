import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import csv

class PL(nn.Module):
    def __init__(self, threshold, n_classes=10, lambda_decay=0.999):
        super().__init__()
        self.th = threshold
        self.n_classes = n_classes
        self.lambda_decay = lambda_decay
        self.min_th = 0.8
        self.max_th = 0.95
        self.iter_count = 0
        self.confidence = 0

    def forward(self, x, y, model, mask):
        # import ipdb; ipdb.set_trace()
        y_probs = y.softmax(1)    
        onehot_label = self.__make_one_hot(y_probs.max(1)[1]).float()
        max_confidence = y_probs.max(dim=1)[0].mean().item()
        
        #if self.iter_count > 700 and self.th > self.min_th:
        from statistics import harmonic_mean
        
        #global threshold
        if max_confidence > self.confidence and self.th >= self.min_th:
            self.th = self.th - ((1-self.lambda_decay) * (max_confidence-self.confidence))
            
         
        #class threshold
        #Add your update rule   
        #self.th_per_class = None
        
        #combined
        #self.final_th = harmonic_mean([self.th, self.th_per_class])
        
        
        
        self.confidence = max_confidence
        
        gt_mask = (y_probs > self.th).float()
        gt_mask = gt_mask.max(1)[0] # reduce_any
        
        lt_mask = 1 - gt_mask # logical not
        p_target = gt_mask[:,None] * 10 * onehot_label + lt_mask[:,None] * y_probs
        # model.update_batch_stats(False)
        output = model(x)
        loss = (-(p_target.detach() * F.log_softmax(output, 1)).sum(1)*mask).mean()
        # model.update_batch_stats(True)
        self.iter_count += 1
        return loss

    def __make_one_hot(self, y):
        return torch.eye(self.n_classes, device=y.device)[y]
