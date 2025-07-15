
import torch
import torch.nn as nn
import torch.nn.functional as F

class DCKDLoss(nn.Module):
    """
    Distillation avec structure et diversité.
    Combine distillation classique avec corrélation inter-canaux.
    """
    def __init__(self, temperature=4.0, alpha=0.5, beta=1.0):
        super(DCKDLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha  # KD loss
        self.beta = beta    # Correlation loss

        # Projection linéaire (sera initialisée dynamiquement au besoin)
        self.projection = None

    def forward(self, student_logits, teacher_logits, student_features, teacher_features):
        # --- KD classique ---
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # --- Corrélation inter-feature ---
        corr_loss = 0.0
        for f_s, f_t in zip(student_features, teacher_features):
            # Aplatir
            f_s = f_s.view(f_s.size(0), -1)
            f_t = f_t.view(f_t.size(0), -1)

            # Adapter la dimension si nécessaire
            if f_s.shape[1] != f_t.shape[1]:
                self.projection = nn.Linear(f_s.shape[1], f_t.shape[1]).to(f_s.device)
                f_s = self.projection(f_s)

            # Cosine similarity
            corr_loss += 1 - F.cosine_similarity(f_s, f_t, dim=1).mean()

        # Perte totale
        total_loss = self.alpha * kd_loss + self.beta * corr_loss
        return total_loss
