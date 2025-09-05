
import torch
import torch.nn as nn
import torch.nn.functional as F

class DCKDLoss(nn.Module):
    """
    Distillation avec structure et diversité.
    Combine distillation classique avec corrélation inter-canaux.
    """
    # def __init__(self, temperature=4.0, alpha=0.5, beta=1.0):
    #   super(DCKDLoss, self).__init__()
    #   self.temperature = temperature
    #   self.alpha = alpha  # KD loss
    #   self.beta = beta    # Correlation loss

        # Projection linéaire (sera initialisée dynamiquement au besoin)
        # self.projection = None
        
    def __init__(self, temperature=4.0, alpha=0.5, beta=2.5, gamma=1.0, delta=0.0, use_diversity=False):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha    # poids KD
        self.beta = beta      # poids corrélation
        self.gamma = gamma    # poids CE (labels durs)
        self.delta = delta    # poids diversité (optionnel à 0 pour l’instant)
        self.use_diversity = use_diversity
       
       # Projection linéaire (sera initialisée dynamiquement au besoin)
        self.projection = None
        self.initialized = False  # ← AJOUTE CETTE LIGNE

    def forward(self, student_logits, teacher_logits, targets, student_features, teacher_features):
        # --- KD classique ---
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 2) CrossEntropy (labels durs)
        cls_loss = F.cross_entropy(student_logits, targets)
        
        # --- Corrélation inter-feature ---
        corr_loss = 0.0
        for f_s, f_t in zip(student_features, teacher_features):
            # Aplatir
            f_s = f_s.view(f_s.size(0), -1)
            f_t = f_t.view(f_t.size(0), -1)

            if not self.initialized:
                if f_s.shape[1] != f_t.shape[1]:
                    self.projection = nn.Linear(f_s.shape[1], f_t.shape[1]).to(f_s.device)
                self.initialized = True

            if self.projection is not None:
                f_s = self.projection(f_s)

            # Cosine similarity
            corr_loss = corr_loss + (1.0 - F.cosine_similarity(f_s, f_t, dim=1).mean())
            
        # 4) Diversité (désactivée pour cette étape)
        div_loss = torch.tensor(0.0, device=student_logits.device)
            
        # 5) Agrégation
        total_loss = self.gamma * cls_loss + self.alpha * kd_loss + self.beta * corr_loss
        if self.use_diversity and self.delta != 0.0:
                total_loss = total_loss + self.delta * div_loss

        logs = {
            'ce': cls_loss.detach(),
            'kd': kd_loss.detach(),
            'corr': torch.as_tensor(corr_loss, device=student_logits.device).detach(),
            'div': div_loss.detach()
        }
        return total_loss, logs
