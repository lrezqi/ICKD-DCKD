
import torch
import torch.nn as nn
import torch.nn.functional as F

class DCKDLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5, beta=2.5, gamma=1.0, delta=0.0, use_diversity=False):
        super().init()
        self.temperature = temperature
        self.alpha = alpha    # poids KD
        self.beta = beta      # poids corrélation
        self.gamma = gamma    # poids CE (labels durs)
        self.delta = delta    # poids diversité (optionnel à 0 pour l’instant)
        self.use_diversity = use_diversity
        self.projection = None
        self.initialized = False 

    @staticmethod
    def _bchw_to_bcn(x):
        # x: BxCxHxW -> BxCxN
        B, C, H, W = x.shape
        return x.view(B, C, H*W)

    @staticmethod
    def _gram_batched(x_bcn):
        # x_bcn: BxCxN ; normaliser L2 sur N pour échelle-invariance puis Gram BxCxC
        x = F.normalize(x_bcn, p=2, dim=2)   # L2 sur l’axe spatial
        return torch.bmm(x, x.transpose(1, 2))  # BxCxC

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
        
        # --- Corr inter-canaux (Gram CxC)
        corr_loss = 0.0
        levels = 0
        for i, (f_s, f_t) in enumerate(zip(student_features, teacher_features)):
            # Aplatir
            if f_s.dim() != 4 or f_t.dim() != 4:
                raise ValueError(f"Expected 4D features at level {i}, got {f_s.shape} and {f_t.shape}")
        
            # Si HxW diffèrent, optionnel: adapter f_s à la spatialité de f_t (interpolation bilinéaire)
            if f_s.shape[2:] != f_t.shape[2:]:
                f_s = F.interpolate(f_s, size=f_t.shape[2:], mode='bilinear', align_corners=False)

            xs = self._bchw_to_bcn(f_s)  # BxCxN
            xt = self._bchw_to_bcn(f_t)  # BxCxN

            Gs = self._gram_batched(xs)  # BxCxC
            Gt = self._gram_batched(xt)  # BxCxC

            diff = Gs - Gt
            B, C, _ = diff.shape
            lc = diff.pow(2).sum() / (B * C * C)
            corr_loss = corr_loss + lc
            levels += 1
            
        if levels > 0:
            corr_loss = corr_loss / levels    
            
        # 4) Diversité (pas encore)
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
