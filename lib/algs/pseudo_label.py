import torch
import torch.nn as nn
import torch.nn.functional as F

class PL(nn.Module):
    def __init__(self, threshold, n_classes=10, freq_per_class=None):
        super().__init__()
        self.th = threshold
        self.n_classes = n_classes
        self.freq_per_class = freq_per_class
        self.eps = 1e-6
        # Calculate threshold per class as 1/((# of samples per class) + eps)
        if self.freq_per_class is not None:
            min_threshold = 0.75
            max_threshold = 0.95
            # Add eps to avoid division by zero
            # Calculate threshold per class as 1/((# of samples per class) + eps). Normalize to [min_threshold, max_threshold]
            # self.th_per_class = 1 / (self.freq_per_class + self.eps)
            self.th_per_class = self.freq_per_class
            self.th_per_class = (self.th_per_class - self.th_per_class.min()) / (self.th_per_class.max() - self.th_per_class.min())
            self.th_per_class = self.th_per_class * (max_threshold - min_threshold) + min_threshold
        else:
            self.th_per_class = None

    def forward(self, x, y, model, mask):
        # import ipdb; ipdb.set_trace()
        y_probs = y.softmax(1)
        onehot_label = self.__make_one_hot(y_probs.max(1)[1]).float()
        if self.th_per_class is not None:
            gt_mask = (y_probs > self.th_per_class).float()
        else:
            gt_mask = (y_probs > self.th).float()
        gt_mask = gt_mask.max(1)[0] # reduce_any
        lt_mask = 1 - gt_mask # logical not
        p_target = gt_mask[:,None] * 10 * onehot_label + lt_mask[:,None] * y_probs
        # model.update_batch_stats(False)
        output = model(x)
        loss = (-(p_target.detach() * F.log_softmax(output, 1)).sum(1)*mask).mean()
        # model.update_batch_stats(True)
        return loss

    def __make_one_hot(self, y):
        return torch.eye(self.n_classes, device=y.device)[y]
