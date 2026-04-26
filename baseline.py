"""
baseline.py
===========
Test-Time Adaptation (TTA) baselines for Video Anomaly Detection.

Implements six TTA algorithms as baselines, all evaluated against the same
VAD task defined in TTA1.py:

  1. TENT   — entropy minimisation over BN affine params (Wang et al., 2021)
  2. EATA   — efficient anti-forgetting TTA with sample weighting (Niu et al., 2022)
  3. CoTTA  — continual TTA with stochastic restore + augmentation (Wang et al., 2022)
  4. SAR    — sharpness-aware, reliable-sample filtering TTA (Niu et al., 2023)
  5. READ   — reliable entropy-based adaptation (Sun et al., 2023)
  6. SUMA   — selective update memory-augmented TTA (Xiao et al., 2023)

Usage
-----
    python baseline.py

The script imports VAD_Model, MIL, Normal_Loader, Anomaly_Loader and the
test_abnormal evaluation loop directly from TTA1.py so all baselines are
evaluated under the exact same data and metric protocol.
"""

import copy
import math
import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Import everything we need from the main model file
# ---------------------------------------------------------------------------
from TTA1 import (
    VAD_Model,
    MIL,
    contrastive_loss,
    focal_loss,
    anomaly_score_loss,
)
from dataset1 import Normal_Loader, Anomaly_Loader

# UCF-Crime class names (same as TTA1.py)
class_names = [
    'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion',
    'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting',
    'Stealing', 'Vandalism'
]
class_to_id = {name: i + 1 for i, name in enumerate(class_names)}


# ===========================================================================
# Shared utility — collect BatchNorm layers (needed by TENT / EATA / CoTTA / SAR)
# ===========================================================================
def collect_bn_params(model):
    """
    Return (params, names) for TTA adaptation.

    For architectures without BN (like this MambaBlock-based VAD_Model),
    we adapt the output head layers (fc + projection) whose parameters sit
    directly on the gradient path from the frozen backbone to the loss.

    Priority order:
      1. BatchNorm / LayerNorm / GroupNorm affine parameters
      2. Output head: fc and projection weight + bias  (VAD_Model specific)
      3. Bias terms of all Linear layers
      4. ALL parameters (last resort)
    """
    params, names = [], []

    # 1 — normalisation affine params
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d,
                               nn.LayerNorm, nn.GroupNorm)):
            if module.weight is not None:
                params.append(module.weight)
                names.append(f"{name}.weight")
            if module.bias is not None:
                params.append(module.bias)
                names.append(f"{name}.bias")
    if params:
        return params, names

    # 2 — output head layers: fc and projection (VAD_Model specific)
    #     These are always on the gradient path regardless of backbone freeze.
    head_names = {'fc', 'projection'}
    for name, module in model.named_modules():
        if name in head_names and isinstance(module, nn.Linear):
            if module.weight is not None:
                params.append(module.weight)
                names.append(f"{name}.weight")
            if module.bias is not None:
                params.append(module.bias)
                names.append(f"{name}.bias")
    if params:
        return params, names

    # 3 — bias terms of all Linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.bias is not None:
            params.append(module.bias)
            names.append(f"{name}.bias")
    if params:
        return params, names

    # 4 — full fallback
    for name, p in model.named_parameters():
        params.append(p)
        names.append(name)
    return params, names


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Shannon entropy of a sigmoid probability.
    Pass 'out' (raw FC logit) NOT 'anomaly_score' (s_t) — s_t loses its
    grad_fn inside AnomalyGate due to in-place buffer ops.
    """
    p = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
    return -(p * p.log() + (1 - p) * (1 - p).log())


# ===========================================================================
# Shared evaluation loop — mirrors test_abnormal in TTA1.py exactly,
# adapted to accept any wrapped model that returns (out, score, *rest).
# ===========================================================================
def evaluate(model_wrapper, anomaly_test_loader, normal_test_loader,
             device, epoch, method_name, best_auc, roc_data):
    """
    Evaluate a TTA-wrapped model on the UCF-Crime test set.
    Returns avg_auc (float).
    Prints per-class AUC and average AUC, exactly as TTA1.py does.
    """
    model_wrapper.eval_mode()          # some wrappers need special eval setup

    auc = 0
    class_scores  = {i: [] for i in range(1, 14)}
    class_labels  = {i: [] for i in range(1, 14)}
    normal_scores = []
    normal_labels = []
    class_fpr_tpr = {i: None for i in range(1, 14)}

    with torch.no_grad():
        for i, (data, data2) in enumerate(
                zip(anomaly_test_loader, normal_test_loader)):

            # --- anomaly video ---
            try:
                inputs, gts, frames, class_id = data
                class_id = (int(class_id.item())
                            if isinstance(class_id, torch.Tensor) else int(class_id))
            except (ValueError, TypeError):
                inputs, gts, frames = data
                file_path = (
                    anomaly_test_loader.dataset.data_list[i].split('|')[0]
                    if hasattr(anomaly_test_loader.dataset, 'data_list') else '')
                class_id = 1
                for cn in class_names:
                    if cn.lower() in file_path.lower():
                        class_id = class_to_id[cn]
                        break

            inputs = inputs.view(-1, inputs.size(-1)).to(device)
            score  = model_wrapper.predict(inputs)          # (N,) sigmoid scores
            score  = score.cpu().detach().numpy()

            score_list = np.zeros(frames[0])
            step       = np.round(np.linspace(0, frames[0] // 16, 33))
            for j in range(32):
                score_list[int(step[j]) * 16:int(step[j + 1]) * 16] = \
                    score[j % len(score)]
            gt_list = np.zeros(frames[0])
            for k in range(len(gts) // 2):
                s = max(0, int(gts[k * 2]) - 1)
                e = min(int(gts[k * 2 + 1]), frames[0])
                gt_list[s:e] = 1

            if 1 <= class_id <= 13:
                class_scores[class_id].extend(score_list)
                class_labels[class_id].extend(gt_list)

            # --- normal video ---
            inputs2, gts2, frames2 = data2
            inputs2 = inputs2.view(-1, inputs2.size(-1)).to(device)
            score2  = model_wrapper.predict(inputs2)
            score2  = score2.cpu().detach().numpy()

            score_list2 = np.zeros(frames2[0])
            step2       = np.round(np.linspace(0, frames[0] // 16, 33))
            for kk in range(32):
                score_list2[int(step2[kk]) * 16:int(step2[kk + 1]) * 16] = \
                    score2[kk % len(score2)]
            gt_list2 = np.zeros(frames2[0])
            normal_scores.extend(score_list2)
            normal_labels.extend(gt_list2)

            score_list3 = np.concatenate((score_list, score_list2))
            gt_list3    = np.concatenate((gt_list, gt_list2))
            if len(np.unique(gt_list3)) > 1:
                fpr, tpr, _ = metrics.roc_curve(gt_list3, score_list3, pos_label=1)
                auc        += metrics.auc(fpr, tpr)

    avg_auc = auc / max(1, len(anomaly_test_loader))
    print(f'[{method_name}] Epoch: {epoch}, AUC: {avg_auc:.4f}, '
          f'Best AUC: {max(best_auc, avg_auc):.4f}')

    # --- per-class AUC ---
    class_aucs = []
    for class_id in range(1, 14):
        if class_scores[class_id]:
            scores = np.concatenate((class_scores[class_id], normal_scores))
            labels = np.concatenate((class_labels[class_id], normal_labels))
            if len(np.unique(labels)) > 1:
                fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
                cauc = metrics.auc(fpr, tpr)
                class_aucs.append(cauc)
                class_fpr_tpr[class_id] = (fpr, tpr)
                print(f'  [{method_name}] {class_names[class_id - 1]}: AUC = {cauc:.4f}')
            else:
                print(f'  [{method_name}] {class_names[class_id - 1]}: AUC = N/A')
        else:
            print(f'  [{method_name}] {class_names[class_id - 1]}: AUC = N/A (no data)')

    if class_aucs:
        avg_class_auc = sum(class_aucs) / len(class_aucs)
        print(f'[{method_name}] Average AUC across classes: {avg_class_auc:.4f}')
        roc_data[epoch] = {
            'average_auc': avg_class_auc,
            'class_fpr_tpr': class_fpr_tpr,
        }
    else:
        print(f'[{method_name}] Average AUC across classes: N/A')

    # save checkpoint if improved
    if avg_auc > best_auc:
        os.makedirs('checkpoint', exist_ok=True)
        state = {'net': model_wrapper.model.state_dict()}
        torch.save(state, f'./checkpoint/ckpt_{method_name}.pth')
        print(f'[{method_name}] Saved new best checkpoint (AUC {avg_auc:.4f})')

    return avg_auc


# ===========================================================================
# Base wrapper — all TTA baselines subclass this
# ===========================================================================
class TTAWrapper:
    """
    Thin wrapper around VAD_Model that:
      • stores the original model weights for stochastic restore (CoTTA)
      • exposes .predict(x) → sigmoid anomaly scores
      • exposes .adapt(x)   → optional per-batch TTA update
      • exposes .eval_mode() → set model to eval
    """
    def __init__(self, model: VAD_Model, device: str):
        self.model  = model
        self.device = device
        # snapshot of initial weights for restore mechanisms
        self._initial_state = copy.deepcopy(model.state_dict())

    def eval_mode(self):
        self.model.eval()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass → sigmoid scores (B,). Uses out (FC logit), same as TTA1.py test_abnormal."""
        out, anomaly_score, proj, L_temp, L_mem, g_t, alpha_t = self.model(x)
        return torch.sigmoid(out.squeeze(1))

    def adapt(self, x: torch.Tensor):
        """Override in subclasses to perform a TTA update step."""
        pass

    def reset(self):
        """Restore model to initial weights."""
        self.model.load_state_dict(self._initial_state)


# ===========================================================================
# 1. TENT — Test Entropy Minimization (Wang et al., ICLR 2021)
#    Minimises prediction entropy by updating only BN affine parameters.
# ===========================================================================
class TENTWrapper(TTAWrapper):
    """
    TENT: minimise H(p) = -p log p - (1-p) log(1-p)
    over BatchNorm / LayerNorm affine parameters only.
    """
    def __init__(self, model: VAD_Model, device: str, lr: float = 1e-3):
        super().__init__(model, device)
        # collect BEFORE freezing so fallback can see all params
        params, _ = collect_bn_params(self.model)
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
        self.optimizer = optim.Adam(params, lr=lr)

    def eval_mode(self):
        # No BN layers in VAD_Model, so keep full train() for adapt steps.
        # predict() already switches to no_grad after adapt.
        self.model.train()

    def adapt(self, x: torch.Tensor):
        self.model.train()
        out, anomaly_score, proj, L_temp, L_mem, g_t, alpha_t = self.model(x)
        logit = out.squeeze(1)
        if not logit.requires_grad:
            return   # nothing to adapt if graph is broken
        entropy = softmax_entropy(logit).mean()
        self.optimizer.zero_grad()
        entropy.backward()
        self.optimizer.step()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.adapt(x)
        with torch.no_grad():
            out, anomaly_score, proj, L_temp, L_mem, g_t, alpha_t = self.model(x)
        return torch.sigmoid(out.squeeze(1))


# ===========================================================================
# 2. EATA — Efficient Anti-forgetting TTA (Niu et al., ICML 2022)
#    Adds (a) sample weighting by entropy and (b) Fisher-regularised
#    anti-forgetting penalty to TENT.
# ===========================================================================
class EATAWrapper(TTAWrapper):
    """
    EATA: entropy-weighted adaptation with Fisher-information anti-forgetting.
      L = Σ_t w_t * H(p_t)  +  lambda_f * Σ_k F_k (theta_k - theta_k^*)^2
    where w_t = 1 - H(p_t)/H_max  (higher-entropy samples get lower weight).
    """
    def __init__(self, model: VAD_Model, device: str,
                 lr: float = 1e-3, lambda_f: float = 2000.0,
                 fisher_clip: float = 1e-3):
        super().__init__(model, device)
        # collect BEFORE freezing
        params, self._param_names = collect_bn_params(self.model)
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
        self._params    = params
        self.optimizer  = optim.Adam(params, lr=lr)
        self.lambda_f   = lambda_f
        self.fisher_clip = fisher_clip
        self._fisher    = [torch.zeros_like(p) for p in params]
        self._anchor    = [p.data.clone() for p in params]
        self._H_max     = math.log(2)

    def _fisher_update(self, loss):
        """Accumulate EMA of squared gradients as Fisher diagonal estimate."""
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        with torch.no_grad():
            for i, p in enumerate(self._params):
                if p.grad is not None:
                    self._fisher[i] = (0.9 * self._fisher[i]
                                       + 0.1 * p.grad.detach().pow(2)
                                       .clamp(max=self.fisher_clip))

    def adapt(self, x: torch.Tensor):
        self.model.train()
        out, anomaly_score, proj, L_temp, L_mem, g_t, alpha_t = self.model(x)

        logit = out.squeeze(1)
        if not logit.requires_grad:
            return
        # use 'out' (raw FC logit) — anomaly_score (s_t) loses grad_fn in AnomalyGate
        H    = softmax_entropy(logit)                        # (B,)
        w    = (1.0 - H / self._H_max).clamp(0.0, 1.0)     # sample weights
        L_e  = (w * H).mean()                               # entropy term

        # anti-forgetting regulariser
        L_af = sum(
            (f * (p - a).pow(2)).sum()
            for f, p, a in zip(self._fisher, self._params, self._anchor)
        )
        loss = L_e + self.lambda_f * L_af

        self._fisher_update(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.adapt(x)
        with torch.no_grad():
            out, anomaly_score, proj, L_temp, L_mem, g_t, alpha_t = self.model(x)
        return torch.sigmoid(out.squeeze(1))


# ===========================================================================
# 3. CoTTA — Continual Test-Time Adaptation (Wang et al., CVPR 2022)
#    Stochastic weight restore + augmentation-averaged predictions.
# ===========================================================================
class CoTTAWrapper(TTAWrapper):
    """
    CoTTA: at each step
      (a) produce augmentation-averaged prediction
      (b) entropy-minimise on affine params
      (c) stochastically restore a fraction (p_restore) of params to anchors
    """
    def __init__(self, model: VAD_Model, device: str,
                 lr: float = 1e-3, p_restore: float = 0.01,
                 n_aug: int = 4, noise_std: float = 0.05):
        super().__init__(model, device)
        # collect BEFORE freezing
        params, _ = collect_bn_params(self.model)
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
        self.optimizer  = optim.Adam(params, lr=lr)
        self.p_restore  = p_restore
        self.n_aug      = n_aug
        self.noise_std  = noise_std
        self._teacher   = copy.deepcopy(model)
        for p in self._teacher.parameters():
            p.requires_grad_(False)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Lightweight feature-space augmentation: additive Gaussian noise."""
        return x + torch.randn_like(x) * self.noise_std

    def _teacher_predict(self, x: torch.Tensor) -> torch.Tensor:
        self._teacher.eval()
        with torch.no_grad():
            out_t, score, _, _, _, _, _ = self._teacher(x)
        return torch.sigmoid(out_t.squeeze(1))

    def _ema_update_teacher(self, alpha: float = 0.999):
        with torch.no_grad():
            for t_p, s_p in zip(self._teacher.parameters(),
                                 self.model.parameters()):
                t_p.data = alpha * t_p.data + (1 - alpha) * s_p.data

    def adapt(self, x: torch.Tensor):
        self.model.train()
        # augmentation-averaged teacher pseudo-label (no grad needed)
        aug_scores = torch.stack(
            [self._teacher_predict(self._augment(x)) for _ in range(self.n_aug)],
            dim=0
        ).mean(dim=0).detach()                               # (B,)

        # student forward
        out, anomaly_score, proj, L_temp, L_mem, g_t, alpha_t = self.model(x)
        # use out (raw FC logit) — anomaly_score loses grad_fn inside AnomalyGate
        student_logit = out.squeeze(1)
        if not student_logit.requires_grad:
            return

        # consistency loss: BCE between student logit and teacher pseudo-label
        L_con = F.binary_cross_entropy_with_logits(
            student_logit, aug_scores.clamp(0, 1)
        )

        self.optimizer.zero_grad()
        L_con.backward()
        self.optimizer.step()

        # EMA teacher update
        self._ema_update_teacher()

        # stochastic weight restore
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if torch.rand(1).item() < self.p_restore:
                    p.data.copy_(self._initial_state[name])

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.adapt(x)
        # augmentation-averaged prediction at test time
        preds = torch.stack(
            [self._teacher_predict(self._augment(x)) for _ in range(self.n_aug)],
            dim=0
        ).mean(dim=0)
        return preds


# ===========================================================================
# 4. SAR — Sharpness-Aware, Reliable-sample TTA (Niu et al., ICLR 2023)
#    Filters unreliable (high-entropy) samples and applies SAM-style update.
# ===========================================================================
class SARWrapper(TTAWrapper):
    """
    SAR: only adapt on samples with entropy below threshold e0.
    Uses Sharpness-Aware Minimisation (SAM) inner / outer step.
      L = mean_reliable H(p)
    """
    def __init__(self, model: VAD_Model, device: str,
                 lr: float = 1e-3, e0: float = 0.4, rho: float = 0.05):
        super().__init__(model, device)
        # collect BEFORE freezing
        params, _ = collect_bn_params(self.model)
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
        self._params    = params
        self.lr         = lr
        self.e0         = e0
        self.rho        = rho
        self.optimizer  = optim.SGD(params, lr=lr, momentum=0.9)

    def _sam_step(self, x: torch.Tensor):
        """One SAM perturbation + update using two forward passes."""
        # --- 1st forward: compute loss and gradients at current weights ---
        self.model.train()
        out, anomaly_score, proj, L_temp, L_mem, g_t, alpha_t = self.model(x)
        logit = out.squeeze(1)
        if not logit.requires_grad:
            return
        H = softmax_entropy(logit)
        reliable = H < self.e0
        if reliable.sum() == 0:
            return
        loss1 = H[reliable].mean()

        self.optimizer.zero_grad()
        loss1.backward()
        grad_norm = torch.norm(
            torch.stack([p.grad.norm() for p in self._params if p.grad is not None])
        )
        scale = self.rho / (grad_norm + 1e-12)

        # perturb weights
        with torch.no_grad():
            for p in self._params:
                if p.grad is not None:
                    p._e_w = p.grad * scale
                    p.data.add_(p._e_w)

        # --- 2nd forward: compute loss at perturbed weights ---
        out2, anomaly_score2, proj2, L_temp2, L_mem2, g_t2, alpha_t2 = self.model(x)
        H2 = softmax_entropy(out2.squeeze(1))
        reliable2 = H2 < self.e0
        if reliable2.sum() == 0:
            # restore and return
            with torch.no_grad():
                for p in self._params:
                    if hasattr(p, '_e_w'):
                        p.data.sub_(p._e_w)
                        del p._e_w
            return
        loss2 = H2[reliable2].mean()

        self.optimizer.zero_grad()
        loss2.backward()

        # restore weights then update
        with torch.no_grad():
            for p in self._params:
                if hasattr(p, '_e_w'):
                    p.data.sub_(p._e_w)
                    del p._e_w
        self.optimizer.step()

    def adapt(self, x: torch.Tensor):
        self._sam_step(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.adapt(x)
        with torch.no_grad():
            out, anomaly_score, proj, L_temp, L_mem, g_t, alpha_t = self.model(x)
        return torch.sigmoid(out.squeeze(1))


# ===========================================================================
# 5. READ — Reliable Entropy-based Adaptation (Sun et al., 2023)
#    Extends TENT with a two-stage reliability check:
#      Stage 1 — reject samples whose entropy > upper threshold H_hi
#      Stage 2 — down-weight samples in the uncertain band [H_lo, H_hi]
#    Uses an EMA prototype of reliable features to anchor the update.
# ===========================================================================
class READWrapper(TTAWrapper):
    """
    READ: reliability-filtered entropy minimisation.
      w_t = 0              if H(p_t) > H_hi   (hard reject)
      w_t = (H_hi - H_t)  if H_lo < H_t < H_hi (soft weight)
      w_t = 1              if H_t < H_lo      (fully reliable)
    L = Σ_t w_t * H(p_t)
    Also maintains an EMA feature prototype of reliable frames.
    """
    def __init__(self, model: VAD_Model, device: str,
                 lr: float = 1e-3,
                 H_lo: float = 0.2, H_hi: float = 0.6,
                 ema_decay: float = 0.99):
        super().__init__(model, device)
        # collect BEFORE freezing
        params, _ = collect_bn_params(self.model)
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
        self.optimizer  = optim.Adam(params, lr=lr)
        self.H_lo       = H_lo
        self.H_hi       = H_hi
        self.ema_decay  = ema_decay
        self._proto     = None

    def _update_proto(self, features: torch.Tensor, reliable_mask: torch.Tensor):
        if reliable_mask.sum() == 0:
            return
        z_rel = features[reliable_mask].detach().mean(dim=0)
        if self._proto is None:
            self._proto = z_rel
        else:
            self._proto = self.ema_decay * self._proto + (1 - self.ema_decay) * z_rel

    def adapt(self, x: torch.Tensor):
        self.model.train()
        out, anomaly_score, proj, L_temp, L_mem, g_t, alpha_t = self.model(x)

        logit = out.squeeze(1)
        if not logit.requires_grad:
            return
        # use 'out' (raw FC logit) — anomaly_score (s_t) loses grad_fn in AnomalyGate
        H = softmax_entropy(logit)                           # (B,)

        # reliability weights
        w = torch.where(
            H > self.H_hi,
            torch.zeros_like(H),
            torch.where(
                H < self.H_lo,
                torch.ones_like(H),
                (self.H_hi - H) / (self.H_hi - self.H_lo + 1e-8)
            )
        )

        reliable_mask = (H < self.H_lo).detach()

        # get features for prototype update (no grad needed)
        with torch.no_grad():
            _, _, proj_det, _, _, _, _ = self.model(x)
        self._update_proto(proj_det, reliable_mask)

        # prototype consistency regulariser
        L_proto = torch.tensor(0.0, device=x.device)
        if self._proto is not None and reliable_mask.sum() > 0:
            z_rel   = proj[reliable_mask]
            proto   = self._proto.detach().unsqueeze(0).expand_as(z_rel)
            L_proto = F.mse_loss(z_rel, proto)

        L = (w * H).mean() + 0.1 * L_proto

        self.optimizer.zero_grad()
        L.backward()
        self.optimizer.step()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.adapt(x)
        with torch.no_grad():
            out, anomaly_score, proj, L_temp, L_mem, g_t, alpha_t = self.model(x)
        return torch.sigmoid(out.squeeze(1))


# ===========================================================================
# 6. SUMA — Selective Update Memory-Augmented TTA (Xiao et al., 2023)
#    Maintains a memory queue of reliable feature–label pairs.
#    Adaptation loss = entropy + memory contrastive alignment term.
# ===========================================================================
class SUMAWrapper(TTAWrapper):
    """
    SUMA: memory-augmented selective TTA.
      Memory queue stores (feature, pseudo_label) of reliable samples.
      L = H(p_t) + lambda_m * L_mem_contrast
    where L_mem_contrast pulls current feature towards similar memory items.
    """
    def __init__(self, model: VAD_Model, device: str,
                 lr: float = 1e-3, queue_size: int = 256,
                 H_thresh: float = 0.4, lambda_m: float = 0.1,
                 temperature: float = 0.07):
        super().__init__(model, device)
        # collect BEFORE freezing
        params, _ = collect_bn_params(self.model)
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
        self.optimizer   = optim.Adam(params, lr=lr)
        self.H_thresh    = H_thresh
        self.lambda_m    = lambda_m
        self.temperature = temperature
        # memory queue: deque of (feature_vector, pseudo_score) tuples
        self._memory: deque = deque(maxlen=queue_size)

    def _memory_contrast_loss(self, proj: torch.Tensor,
                               anomaly_score: torch.Tensor) -> torch.Tensor:
        """
        Contrastive alignment: pull current projection toward memory items
        with similar anomaly score (within 0.2), push away from dissimilar.
        """
        if len(self._memory) < 4:
            return torch.tensor(0.0, device=proj.device)

        mem_feats  = torch.stack([m[0] for m in self._memory]).to(proj.device)   # (M, 128)
        mem_scores = torch.stack([m[1] for m in self._memory]).to(proj.device)   # (M,)

        proj_norm = F.normalize(proj, dim=1)              # (B, 128)
        mem_norm  = F.normalize(mem_feats, dim=1)         # (M, 128)
        sim       = torch.matmul(proj_norm, mem_norm.T) / self.temperature  # (B, M)

        cur_s  = anomaly_score.detach().unsqueeze(1)      # (B, 1)
        mem_s  = mem_scores.unsqueeze(0)                  # (1, M)
        pos_mask = (cur_s - mem_s).abs() < 0.2            # (B, M) — similar score → positive

        loss = torch.tensor(0.0, device=proj.device)
        for b in range(proj.size(0)):
            pos = pos_mask[b]
            if pos.sum() == 0 or (~pos).sum() == 0:
                continue
            # InfoNCE-style: positives in numerator, all others in denominator
            log_num = torch.logsumexp(sim[b][pos], dim=0)
            log_den = torch.logsumexp(sim[b],       dim=0)
            loss    = loss - (log_num - log_den)
        return loss / max(proj.size(0), 1)

    def _update_memory(self, proj: torch.Tensor, anomaly_score: torch.Tensor,
                        H: torch.Tensor):
        reliable = H < self.H_thresh
        with torch.no_grad():
            for idx in reliable.nonzero(as_tuple=True)[0]:
                self._memory.append((
                    proj[idx].detach().cpu(),
                    torch.sigmoid(anomaly_score[idx]).detach().cpu()
                ))

    def adapt(self, x: torch.Tensor):
        self.model.train()
        out, anomaly_score, proj, L_temp, L_mem, g_t, alpha_t = self.model(x)

        logit = out.squeeze(1)
        if not logit.requires_grad:
            return
        # use 'out' (raw FC logit) — anomaly_score (s_t) loses grad_fn in AnomalyGate
        H       = softmax_entropy(logit)
        L_ent   = H.mean()
        L_mc    = self._memory_contrast_loss(proj, torch.sigmoid(anomaly_score.detach()))
        loss    = L_ent + self.lambda_m * L_mc

        self._update_memory(proj, anomaly_score, H.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.adapt(x)
        with torch.no_grad():
            out, anomaly_score, proj, L_temp, L_mem, g_t, alpha_t = self.model(x)
        return torch.sigmoid(out.squeeze(1))


# ===========================================================================
# Factory — build a fresh copy of VAD_Model + load checkpoint
# ===========================================================================
def build_model(device: str, input_dim: int = 2048,
                memory_size: int = 128) -> VAD_Model:
    model = VAD_Model(input_size=input_dim, memory_size=memory_size).to(device)
    for ckpt_path in [
        './checkpoint/ckpt_auc_0.8585.pth',
        './checkpoint/ckpt_auc_0.8569.pth',
        './checkpoint/ckpt.pth',
    ]:
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, weights_only=False)
            state_dict = checkpoint['net']
            model_sd   = model.state_dict()
            compat_sd  = {k: v for k, v in state_dict.items()
                          if k in model_sd and v.shape == model_sd[k].shape}
            model_sd.update(compat_sd)
            model.load_state_dict(model_sd)
            print(f'Loaded checkpoint: {ckpt_path}')
            break
    return model


# ===========================================================================
# Main — run all six TTA baselines sequentially
# ===========================================================================
if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    modality   = 'TWO'
    input_dim  = 2048
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_epochs   = 3        # TTA runs a small number of adaptation epochs

    # Data loaders (same as TTA1.py)
    normal_test_dataset   = Normal_Loader(is_train=0, modality=modality)
    anomaly_test_dataset  = Anomaly_Loader(is_train=0, modality=modality)
    normal_test_loader    = DataLoader(normal_test_dataset,  batch_size=1, shuffle=False)
    anomaly_test_loader   = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=False)

    # Baselines to evaluate: (name, wrapper_class, extra_kwargs)
    baselines = [
        ('TENT',  TENTWrapper,  dict(lr=1e-3)),
        ('EATA',  EATAWrapper,  dict(lr=1e-3, lambda_f=2000.0)),
        ('CoTTA', CoTTAWrapper, dict(lr=1e-3, p_restore=0.01, n_aug=4)),
        ('SAR',   SARWrapper,   dict(lr=1e-3, e0=0.4, rho=0.05)),
        ('READ',  READWrapper,  dict(lr=1e-3, H_lo=0.2, H_hi=0.6)),
        ('SUMA',  SUMAWrapper,  dict(lr=1e-3, queue_size=256, H_thresh=0.4)),
    ]

    all_results = {}   # method_name -> best_auc

    for method_name, WrapperClass, kwargs in baselines:
        print(f'\n{"=" * 60}')
        print(f'  Baseline: {method_name}')
        print(f'{"=" * 60}')

        # Fresh model copy for each baseline (no cross-contamination)
        model   = build_model(device, input_dim)
        wrapper = WrapperClass(model, device, **kwargs)

        best_auc = 0.0
        roc_data = {}

        for epoch in range(n_epochs):
            auc = evaluate(
                wrapper,
                anomaly_test_loader,
                normal_test_loader,
                device,
                epoch,
                method_name,
                best_auc,
                roc_data,
            )
            if auc > best_auc:
                best_auc = auc

        all_results[method_name] = best_auc
 
 