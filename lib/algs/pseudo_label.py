import torch
import torch.nn as nn
import torch.nn.functional as F

class PL(nn.Module):
    def __init__(self, threshold, n_classes=10, freq_per_class=None, lambda_decay=0.99):
        super().__init__()
        # self.th = threshold
        # self.th = 1 / n_classes
        self.n_classes = n_classes
        self.freq_per_class = freq_per_class
        self.eps = 1e-6
        self.lambda_decay = lambda_decay
        self.min_th = 0.8
        self.max_th = 0.95
        self.iter_count = 0
        self.confidence = 0
        # Set threshold per class as 1/n_classes
        self.th_per_class = torch.ones(n_classes) / n_classes
        self.th_per_class = self.th_per_class.to('mps')

        # Calculate threshold per class as 1/((# of samples per class) + eps)
        if self.freq_per_class is not None:
            # Add eps to avoid division by zero
            # Calculate threshold per class as 1/((# of samples per class) + eps). Normalize to [min_threshold, max_threshold]
            # self.th_per_class = 1 / (self.freq_per_class + self.eps)
            self.th_per_class = self.freq_per_class
            self.th_per_class = (self.th_per_class - self.th_per_class.min()) / (self.th_per_class.max() - self.th_per_class.min())
            self.th_per_class = self.th_per_class * (self.max_th - self.min_th) + self.min_th
        else:
            self.th_per_class = None

    def forward(self, x, y, model, mask):
        # import ipdb; ipdb.set_trace()
        y_probs = y.softmax(1)
        onehot_label = self.__make_one_hot(y_probs.max(1)[1]).float()

        max_confidence = y_probs.max(dim=1)[0].mean().item()
        
        # if self.iter_count > 700 and self.th > self.min_th:
            # if max_confidence > self.confidence:
        # self.th = self.th * self.lambda_decay + ((1-self.lambda_decay) * (max_confidence))
        self.th_per_class = self.th_per_class * self.lambda_decay + (1. - self.lambda_decay) * y_probs.mean(dim=0)
        # Set final threshold as maxnorm of threshold per class * global threshold
        self.th_per_class = (self.th_per_class - self.th_per_class.min()) / (self.th_per_class.max() - self.th_per_class.min())
        self.th_per_class = self.th_per_class * (self.max_th - self.min_th) + self.min_th

        # self.th_per_class = self.th_per_class * self.th
        # self.confidence = max_confidence


        # Set threshold as MaxNorm of threshold per class * global threshold
        # Place threshold per class in device
        if self.th_per_class is not None:
            gt_mask = (y_probs > self.th_per_class).float()
        else:
            gt_mask = (y_probs > self.th).float()

        gt_mask, gt_mask_idxs = gt_mask.max(1) # reduce_any 

        class_freq = torch.zeros(self.n_classes, device=x.device)
        for i in gt_mask_idxs:
            class_freq[i] += 1


        lt_mask = 1 - gt_mask # logical not
        p_target = gt_mask[:,None] * 10 * onehot_label + lt_mask[:,None] * y_probs
        # model.update_batch_stats(False)
        output = model(x)
        loss = (-(p_target.detach() * F.log_softmax(output, 1)).sum(1)*mask).mean()
        # model.update_batch_stats(True)
        return loss, class_freq

    def __make_one_hot(self, y):
        return torch.eye(self.n_classes, device=y.device)[y]
