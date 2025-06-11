import torch
import torch.nn as nn
import torch.nn.functional as F

class DCKDLoss(nn.Module):
    """
    Dynamic Correlation-based Knowledge Distillation Loss.
    Combine traditional KD with inter-layer correlation and dynamic weighting.
    """
    def __init__(self, temperature=4.0, alpha=0.5, beta=1.0):
        super(DCKDLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha  # For KD loss
        self.beta = beta    # For correlation loss

    def forward(self, student_logits, teacher_logits, student_features, teacher_features):
        # --- Classic KD ---
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # --- Correlation loss (simple cosine similarity between selected layers) ---
        corr_loss = 0
        for f_s, f_t in zip(student_features, teacher_features):
            f_s = F.normalize(f_s.view(f_s.size(0), -1), dim=1)
            f_t = F.normalize(f_t.view(f_t.size(0), -1), dim=1)
            corr_loss += 1 - F.cosine_similarity(f_s, f_t, dim=1).mean()

        total_loss = self.alpha * kd_loss + self.beta * corr_loss
        return total_loss
