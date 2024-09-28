import torch

class MultiClassAccuracyMetrics(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.criteria = kwargs["criteria"]
        self.num_correct = 0
        self.num_total = 0

    def reset(self):
        self.num_correct = 0
        self.num_total = 0

    @torch.inference_mode()
    def update(self, preds, targets):
        preds = torch.argmax(torch.stack(preds, dim=1), dim=2)

        #Hamming
        if self.criteria=='hamming':
            num_correct = (preds == targets).sum()
            num_total = torch.tensor(targets.numel(), device=targets.device)
        else:
        #Exact Match
            num_correct = torch.all(preds == targets, dim=1).sum()
            num_total = torch.tensor(targets.shape[0], device=targets.device)
        
        self.num_correct += num_correct
        self.num_total += num_total
        return float(num_correct) / num_total

    @torch.inference_mode()
    def compute(self):
        return float(self.num_correct) / self.num_total
    
    @torch.no_grad()
    def forward(self, preds, targets):
        return self.update(preds=preds, targets=targets)


class MultiLabelAccuracyMetrics(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.criteria = kwargs["criteria"]
        self.threshold = kwargs["threshold"]
        self.num_correct = 0
        self.num_total = 0

    def reset(self):
        self.num_correct = 0
        self.num_total = 0
    
    @torch.inference_mode()
    def update(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = torch.where(preds < self.threshold, 0, 1)
        #Hamming
        if self.criteria=='hamming':
            num_correct = (preds == targets).sum()
            num_total = torch.tensor(targets.numel(), device=targets.device)
        else:
        #Exact Match
            num_correct = torch.all(preds == targets, dim=1).sum()
            num_total = torch.tensor(targets.shape[0], device=targets.device)
        
        self.num_correct += num_correct
        self.num_total += num_total
        return float(num_correct) / num_total

    @torch.inference_mode()
    def compute(self):
        return float(self.num_correct) / self.num_total
    
    @torch.no_grad()
    def forward(self, preds, targets):
        return self.update(preds=preds, targets=targets)