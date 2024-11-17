import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import csv

class PL(nn.Module):
    def __init__(self, threshold, n_classes=10, lambda_decay=0.5, mu=4):
        super().__init__()
        self.th = threshold
        self.n_classes = n_classes
        self.lambda_decay = 0.99
        self.mu = mu
        self.min_th = 0.6
        self.max_th = 0.95
        self.iter_count = 0
        self.confidence = 0

    def forward(self, x, y, model, mask):
        # import ipdb; ipdb.set_trace()
        y_probs = y.softmax(1)
                
        onehot_label = self.__make_one_hot(y_probs.max(1)[1]).float()
        #print("old thres:", self.th)
        
        max_q_b = y_probs.max(dim=1)[0]
        
        #storing into a csv
        mean_confidence = y_probs.mean(dim=1)[0].mean().item()
        max_confidence = y_probs.max(dim=1)[0].mean().item()
        
        file_path = 'y_probs.csv'
        new_row = [self.th, mean_confidence, max_confidence]

        #print("new row added:", new_row)
        # if max_confidence > self.confidence:
        #     if self.th > self.min_th and self.th < self.max_th:
        #         self.th = self.th - ((1-self.lambda_decay) * max_q_b.mean()).item()
        
        self.confidence = max_q_b.mean().item()
        new_th = (self.lambda_decay * self.th) + ((1-self.lambda_decay) * max_q_b.mean()).item()
        if self.iter_count > 200 and self.iter_count < 3000:
            self.th = new_th
        #if new_th > self.th:
            #self.th = new_th
            #self.th = (self.lambda_decay * self.th) - ((1-self.lambda_decay) * max_q_b.mean()).item()
        #Keep in min-max range
        # if self.th < self.min_th:
        #     self.th = self.min_th
        # if self.th > self.max_th:
        #     self.th = self.max_th
        gt_mask = (y_probs > self.th).float()
        gt_mask = gt_mask.max(1)[0] # reduce_any
        
        lt_mask = 1 - gt_mask # logical not
        p_target = gt_mask[:,None] * 10 * onehot_label + lt_mask[:,None] * y_probs
        # model.update_batch_stats(False)
        output = model(x)
        pred_probs = F.softmax(output, 1)
        pred_onehot_label = torch.zeros_like(pred_probs).scatter_(1, pred_probs.argmax(dim=1, keepdim=True), 1)

        pred_classes = pred_onehot_label.argmax(dim=1)
        true_classes = onehot_label.argmax(dim=1)
        #print(y_probs)
        correct_predictions = (pred_classes == true_classes).sum().item()
        total_images = onehot_label.size(0)
        #print(f"Correctly classified images: {correct_predictions} out of {total_images}")
        new_row = new_row + [gt_mask.sum().item(), correct_predictions, total_images]
        if not os.path.isfile(file_path):
            with open(file_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(new_row)
        else:
            with open(file_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(new_row)
        loss = (-(p_target.detach() * F.log_softmax(output, 1)).sum(1)*mask).mean()
        # model.update_batch_stats(True)
        ##quit()
        self.iter_count += 1
        return loss

    def __make_one_hot(self, y):
        return torch.eye(self.n_classes, device=y.device)[y]
