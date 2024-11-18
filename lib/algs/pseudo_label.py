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
        self.min_th = 0.8
        self.max_th = 0.95
        self.iter_count = 0
        self.confidence = 0
        self.pseudo_thres = threshold

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

        #CONFIG1
        #self.th = (self.lambda_decay * self.th) + ((1-self.lambda_decay) * max_q_b.mean()).item()

        #CONFIG2
        # new_th = (self.lambda_decay * self.th) + ((1-self.lambda_decay) * max_q_b.mean()).item()
        # if new_th > self.th:
        #     self.th = new_th
        
        #CONFIG3
        # self.th = self.th
        
        #CONFIG 5
        # new_th = (self.lambda_decay * self.th) + ((1-self.lambda_decay) * max_q_b.mean()).item()
        # if new_th > self.th and self.iter > 2000:
        #     self.th = new_th
        
        #CONFIG 6
        # new_th = (self.lambda_decay * self.th) + ((1-self.lambda_decay) * max_q_b.mean()).item()
        # if self.iter_count > 200 and self.iter_count < 3000:
        #     self.th = new_th
        
        #CONFIG 7
        # new_th = (self.lambda_decay * self.pseudo_thres) + ((1-self.lambda_decay) * max_q_b.mean()).item()
        # if new_th > self.pseudo_thres:
        #     #maybe remove the lamda decay for next config
        #     self.th = (self.lambda_decay * self.th) - ((1-self.lambda_decay) * max_q_b.mean()).item()
        #     self.pseudo_thres = new_th
            
        #CONFIG 8
        # if self.iter_count > 700:
        #     self.th = self.th - 0.00001
            
        #CONFIG 9
        # if self.iter_count > 700 and self.th > self.min_th:
        #     self.th = self.th - 0.00001
        #this one works well
        
        #CONFIG 10
        if self.iter_count > 700 and self.th > self.min_th:
            if max_confidence > self.confidence:
                self.th = self.th - (0.0001 * (max_confidence-self.confidence))
        
        self.confidence = max_confidence
        
        
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
