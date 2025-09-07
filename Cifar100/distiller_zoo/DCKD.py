
import torch
import torch.nn as nn
import torch.nn.functional as F

class DCKDLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5, beta=2.5, gamma=1.0, delta=0.0, use_diversity=False):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha    # poids KD
        self.beta = beta      # poids corrélation
        self.gamma = gamma    # poids CE (labels durs)
        self.delta = delta    # poids diversité (optionnel à 0 pour l’instant)
        self.use_diversity = use_diversity
        self.projection = None
        self.initialized = False 

    def forward(self, student_logits, teacher_logits, targets, student_features, teacher_features):
        # Vérification des niveaux de features
        if len(student_features) != len(teacher_features):
            raise ValueError(f"Mismatch feature levels: {len(student_features)} vs {len(teacher_features)}")
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
        levels = 0
        for i, (f_s, f_t) in enumerate(zip(student_features, teacher_features)):
            # Aplatir
            if f_s.dim() != 4 or f_t.dim() != 4:
                raise ValueError(f"Expected 4D features at level {i}, got {f_s.shape} and {f_t.shape}")
            levels += 1
            
            f_s = f_s.view(f_s.size(0), -1)
            f_t = f_t.view(f_t.size(0), -1)
                
            if not self.initialized:
               # f_s et f_t sont BxD après flatten; on aligne seulement la dimension D
                in_feat = f_s.shape[1]   # entier: dimension features student après flatten
                out_feat = f_t.shape[1]  # entier: dimension features teacher après flatten
                if in_feat != out_feat:
                    self.projection = nn.Linear(in_feat, out_feat).to(f_s.device)
                self.initialized = True

            if self.projection is not None:
                f_s = self.projection(f_s)

            # Cosine similarity
            corr_loss = corr_loss + (1.0 - F.cosine_similarity(f_s, f_t, dim=1).mean())
            
        if levels > 0:
            corr_loss = corr_loss / levels  
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
