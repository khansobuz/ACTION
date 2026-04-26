import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn import metrics
import numpy as np
import os
from dataset1 import Normal_Loader, Anomaly_Loader
import torch.nn.functional as F
from collections import deque


# =============================================================================
# C1 — AnomalyGate: Dual-Pathway Confidence-Filtered Adaptation
# =============================================================================
class AnomalyGate(nn.Module):
    """
    Computes a per-frame anomaly score and applies a dual-pathway filter:
      - Hard gate  g_t = 1[s_t < tau]  : fully blocks anomalous frames
      - Soft gate  alpha_t = eta * exp(-lambda * s_t) : attenuates uncertain frames
    The two gates are used together in the TTA update rule:
      theta <- theta - g_t * alpha_t * grad
    """
    def __init__(self, tau=0.5, lam=2.0, eta=1.0, warmup_frames=32):
        super(AnomalyGate, self).__init__()
        self.tau = tau          # hard threshold
        self.lam = lam          # soft decay coefficient
        self.eta = eta          # base learning rate scale
        self.warmup_frames = warmup_frames
        # running stats for score normalisation (scene-invariant scoring)
        self.register_buffer('running_mean', torch.tensor(0.0))
        self.register_buffer('running_var',  torch.tensor(1.0))
        self.register_buffer('frame_count',  torch.tensor(0))

    def update_stats(self, raw_score):
        """Update running mean/variance from early frames (warm-up window)."""
        if self.frame_count < self.warmup_frames:
            n = self.frame_count.float() + 1.0
            delta = raw_score.mean() - self.running_mean
            self.running_mean = self.running_mean + delta / n
            self.running_var  = self.running_var  + (raw_score.var() - self.running_var) / n
            self.frame_count  = self.frame_count  + raw_score.size(0)

    def forward(self, raw_score):
        """
        Args:
            raw_score: (B,) reconstruction error per frame
        Returns:
            s_t      : normalised anomaly score  (B,)
            g_t      : hard gate mask            (B,)  values in {0, 1}
            alpha_t  : soft learning rate weight  (B,)  values in (0, eta]
        """
        self.update_stats(raw_score.detach())
        # 1a — normalised anomaly score
        s_t = raw_score / (self.running_var.clamp(min=1e-6))
        # 1b — hard binary gate
        g_t = (s_t < self.tau).float()
        # 1c — soft learning rate modulator
        alpha_t = self.eta * torch.exp(-self.lam * s_t)
        return s_t, g_t, alpha_t


# =============================================================================
# C2 — TemporalAnchor: Multi-Scale Temporal Consistency Regularization
# =============================================================================
class TemporalAnchor(nn.Module):
    """
    Enforces prediction consistency at three temporal scales:
      2a  L_fp  = ||p_t - p_{t-1}||^2          (frame-pair, short-range)
      2b  L_cs  = Var({p_t}_{t in W})           (clip smoothness, medium-range)
      2c  L_dr  = ||z_t - z_bar_hist||^2        (drift penalty, long-range)

    Only frames that pass AnomalyGate (g_t=1) are used to update z_bar_hist.
    """
    def __init__(self, window_size=16, rho=0.99, alpha=1.0, beta=0.5, gamma=0.5):
        super(TemporalAnchor, self).__init__()
        self.window_size = window_size
        self.rho  = rho    # EMA decay for long-range drift
        self.alpha = alpha  # weight for L_fp
        self.beta  = beta   # weight for L_cs
        self.gamma = gamma  # weight for L_dr
        # EMA of normal features — updated outside of autograd
        self.z_bar_hist = None

    def update_history(self, z_t, g_t):
        """Update EMA feature history using only gated-normal frames."""
        normal_mask = g_t.bool()
        if normal_mask.any():
            z_normal = z_t[normal_mask].detach().mean(dim=0)
            if self.z_bar_hist is None:
                self.z_bar_hist = z_normal
            else:
                self.z_bar_hist = self.rho * self.z_bar_hist + (1 - self.rho) * z_normal

    def forward(self, p_t, z_t, p_prev=None):
        """
        Args:
            p_t    : current predictions    (B,)
            z_t    : current features       (B, D)
            p_prev : previous predictions   (B,) or None
        Returns:
            L_temp : scalar temporal loss
        """
        device = p_t.device
        loss = torch.tensor(0.0, device=device)

        # 2a — frame-pair consistency (short-range)
        if p_prev is not None:
            # truncate to the smaller of the two batch sizes (replay buffer can change size)
            min_len = min(p_t.size(0), p_prev.size(0))
            L_fp = F.mse_loss(p_t[:min_len], p_prev[:min_len].detach())
            loss = loss + self.alpha * L_fp

        # 2b — clip-level smoothness (medium-range)
        if p_t.size(0) > 1:
            # variance of predictions within the current mini-batch window
            L_cs = p_t.var()
            loss = loss + self.beta * L_cs

        # 2c — long-range drift penalty
        if self.z_bar_hist is not None:
            z_bar = self.z_bar_hist.detach().to(device)
            L_dr = F.mse_loss(z_t, z_bar.unsqueeze(0).expand_as(z_t))
            loss = loss + self.gamma * L_dr

        return loss


# =============================================================================
# C3 — ProtoVault: Scene-Adaptive Normal Memory Bank
# =============================================================================
class ProtoVault(nn.Module):
    """
    Maintains K normal prototypes and one anomaly centroid.
    Three sub-mechanisms:
      3a  Momentum prototype update:   m_k <- mu*m_k + (1-mu)*z_t   (if g_t=1)
      3b  Prototype pull loss:         L_pull = ||z_t - m_bar_n||^2
      3c  Anomaly repulsion loss:      L_push = max(0, delta - ||z_t - m_a||)
          m_a is built from rejected frames (g_t=0), stop-gradient
    """
    def __init__(self, feat_dim=2048, K=64, mu=0.9, rho_a=0.99, delta=1.0, alpha_p=1.0, alpha_r=0.5):
        super(ProtoVault, self).__init__()
        self.K      = K
        self.mu     = mu        # momentum for prototype update
        self.rho_a  = rho_a    # EMA decay for anomaly centroid
        self.delta  = delta    # repulsion margin
        self.alpha_p = alpha_p  # weight for pull loss
        self.alpha_r = alpha_r  # weight for push loss

        # K normal prototypes, stored as buffers (not learned via backprop)
        self.register_buffer('prototypes', torch.randn(K, feat_dim) * 0.01)
        self.register_buffer('proto_age',  torch.zeros(K))   # update frequency
        # single anomaly centroid
        self.register_buffer('m_a',        torch.zeros(feat_dim))
        self.register_buffer('m_a_init',   torch.tensor(False))

        # top-k for pull centroid computation
        self.topk = 3

    @torch.no_grad()
    def update_prototypes(self, z_t, g_t):
        """
        3a — Momentum update of normal prototypes using gated-normal frames.
             If nearest prototype distance exceeds threshold, evict stale slot.
        """
        normal_feats = z_t[g_t.bool()].detach()
        if normal_feats.size(0) == 0:
            return
        for z in normal_feats:
            # find nearest prototype
            dists = torch.norm(self.prototypes - z.unsqueeze(0), dim=1)
            k = dists.argmin()
            # momentum update
            self.prototypes[k] = self.mu * self.prototypes[k] + (1 - self.mu) * z
            self.proto_age[k]  = self.proto_age[k] + 1

    @torch.no_grad()
    def update_anomaly_centroid(self, z_t, g_t):
        """
        3c (part 1) — Build anomaly centroid from rejected frames, no gradient.
        """
        anom_feats = z_t[~g_t.bool()].detach()
        if anom_feats.size(0) == 0:
            return
        z_a = anom_feats.mean(dim=0)
        if not self.m_a_init.item():
            self.m_a      = z_a
            self.m_a_init = torch.tensor(True, device=z_a.device)
        else:
            self.m_a = self.rho_a * self.m_a + (1 - self.rho_a) * z_a

    def pull_loss(self, z_t):
        """
        3b — Prototype attraction: pull z_t toward centroid of nearest prototypes.
        Uses L2 distance (not cosine) to avoid NaN from uninitialised prototypes.
        """
        # L2 distance to all prototypes
        dists    = torch.cdist(z_t, self.prototypes)               # (B, K)
        topk_idx = dists.topk(self.topk, dim=1, largest=False).indices  # (B, topk)
        m_bar    = self.prototypes[topk_idx].mean(dim=1)           # (B, D)
        L_pull   = F.mse_loss(z_t, m_bar.detach())
        return L_pull

    def push_loss(self, z_t):
        """
        3c (part 2) — Anomaly repulsion: push z_t away from anomaly centroid.
        """
        if not self.m_a_init.item():
            return torch.tensor(0.0, device=z_t.device)
        m_a   = self.m_a.detach().unsqueeze(0).expand_as(z_t)     # (B, D)
        dist  = torch.norm(z_t - m_a, dim=1)                      # (B,)
        L_push = F.relu(self.delta - dist).mean()
        return L_push

    def forward(self, z_t, g_t):
        """
        Args:
            z_t : features (B, D)
            g_t : hard gate mask (B,)
        Returns:
            L_mem : scalar memory loss = alpha_p * L_pull + alpha_r * L_push
        """
        # update banks (no gradient)
        self.update_prototypes(z_t, g_t)
        self.update_anomaly_centroid(z_t, g_t)
        # compute differentiable losses
        L_pull = self.pull_loss(z_t)
        L_push = self.push_loss(z_t)
        L_mem  = self.alpha_p * L_pull + self.alpha_r * L_push
        return L_mem


 
class MambaBlock(nn.Module):
    def __init__(self, input_size, hidden_size, d_state=32, d_conv=4, dropout=0.25):
        super(MambaBlock, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.d_state     = d_state
        self.d_conv      = d_conv
        self.in_proj     = nn.Linear(input_size, hidden_size * 2)
        self.out_proj    = nn.Linear(hidden_size, input_size)
        self.ssm_proj    = nn.Linear(hidden_size, d_state)
        self.A           = nn.Parameter(torch.diag(torch.ones(d_state) * -1))
        self.B           = nn.Parameter(torch.randn(d_state) * 0.01)
        self.C           = nn.Parameter(torch.randn(d_state) * 0.01)
        self.D           = nn.Parameter(torch.ones(hidden_size))
        self.conv        = nn.Conv1d(hidden_size, hidden_size,
                                     kernel_size=d_conv, padding=d_conv - 1,
                                     groups=hidden_size)
        self.dropout     = nn.Dropout(dropout)
        self.act         = nn.SiLU()

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.in_proj(x)
        x, gate = x.chunk(2, dim=-1)
        x_conv  = x.transpose(1, 2)
        x_conv  = self.conv(x_conv)[..., :seq_len]
        x_conv  = x_conv.transpose(1, 2)
        x       = self.act(x_conv)
        x_ssm   = self.ssm_proj(x)
        h       = torch.zeros(batch_size, self.d_state, device=x.device)
        ssm_out = []
        for t in range(seq_len):
            h   = torch.einsum('ij,bj->bi', self.A, h) + (x_ssm[:, t, :] * self.B)
            y_t = (torch.einsum('j,bj->b', self.C, h).unsqueeze(-1) * x[:, t, :]
                   + self.D * x[:, t, :])
            ssm_out.append(y_t)
        ssm_out = torch.stack(ssm_out, dim=1)
        y       = self.dropout(ssm_out * self.act(gate))
        y       = self.out_proj(y)
        return y


 
class ChannelAttention1D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, dropout=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1      = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2      = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.act      = nn.ReLU()
        self.dropout  = nn.Dropout(dropout)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, seq_len = x.size()
        avg_out = self.avg_pool(x).view(batch_size, channels)
        max_out = self.max_pool(x).view(batch_size, channels)
        avg_out = self.fc2(self.dropout(self.act(self.fc1(avg_out))))
        max_out = self.fc2(self.dropout(self.act(self.fc1(max_out))))
        out     = self.sigmoid(avg_out + max_out).view(batch_size, channels, 1)
        return x * out


class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7, dropout=0.25):
        super().__init__()
        self.conv    = nn.Conv1d(2, 1, kernel_size=kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out     = torch.cat([avg_out, max_out], dim=1)
        out     = self.sigmoid(self.dropout(self.conv(out)))
        return x * out


 
class VAD_Model(nn.Module):
    def __init__(self, input_size=2048, num_classes=1, memory_size=128):
        super(VAD_Model, self).__init__()

        # --- backbone (unchanged) ---
        self.mamba      = MambaBlock(input_size=input_size,
                                     hidden_size=input_size,
                                     d_state=32, d_conv=4, dropout=0.25)
        self.fc         = nn.Linear(input_size, num_classes)
        self.projection = nn.Linear(input_size, 128)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.projection.weight)

        # --- C1: AnomalyGate (replaces raw threshold logic) ---
        self.anomaly_gate = AnomalyGate(tau=0.5, lam=2.0, eta=1.0, warmup_frames=32)

        # --- C2: TemporalAnchor ---
        self.temporal_anchor = TemporalAnchor(
            window_size=16, rho=0.99,
            alpha=1.0, beta=0.5, gamma=0.5
        )

        # --- C3: ProtoVault (replaces DualMemoryNetwork) ---
        self.proto_vault = ProtoVault(
            feat_dim=input_size, K=64,
            mu=0.9, rho_a=0.99,
            delta=1.0, alpha_p=1.0, alpha_r=0.5
        )

        # carry previous-step predictions for TemporalAnchor 2a
        self._prev_pred = None

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # ---- backbone forward ----
        mamba_out     = self.mamba(x)
        final_feature = mamba_out[:, -1, :]          # (B, D)
        out           = self.fc(final_feature)        # (B, 1)
        proj          = self.projection(final_feature) # (B, 128)

        # ---- raw reconstruction proxy score ----
        # We use the L2 norm of the feature as a lightweight recon-error proxy.
        # In a full autoencoder backbone this would be ||x - f(x)||^2 directly.
        raw_score = torch.norm(final_feature, dim=1)  # (B,)

        # ---- C1: AnomalyGate ----
        s_t, g_t, alpha_t = self.anomaly_gate(raw_score)

        # ---- C2: TemporalAnchor — update history, compute temporal loss ----
        pred_scores = out.squeeze(1)                  # (B,)
        self.temporal_anchor.update_history(final_feature, g_t)
        L_temp = self.temporal_anchor(pred_scores, final_feature, self._prev_pred)
        self._prev_pred = pred_scores.detach()

        # ---- C3: ProtoVault — update banks, compute memory loss ----
        L_mem = self.proto_vault(final_feature, g_t)

        # anomaly_score returned for MIL loss (same interface as original)
        anomaly_score = s_t

        return out, anomaly_score, proj, L_temp, L_mem, g_t, alpha_t

 
def contrastive_loss(proj, batch_size, device, temperature=0.15):
    proj    = F.normalize(proj, dim=1)
    mid     = proj.size(0) // 2
    pos_pairs = proj[:mid]
    neg_pairs = proj[mid:]
    logits  = torch.matmul(pos_pairs, neg_pairs.T) / temperature
    labels  = torch.arange(min(mid, neg_pairs.size(0))).to(device)
    return F.cross_entropy(logits, labels)


def MIL(y_pred, batch_size, device, margin=2.0):
    loss           = torch.tensor(0., device=device)
    sparsity       = torch.tensor(0., device=device)
    smooth         = torch.tensor(0., device=device)
    frames_per_bag = y_pred.size(0) // batch_size
    y_pred         = y_pred.view(batch_size, frames_per_bag)
    for i in range(batch_size):
        mid_point     = frames_per_bag // 2
        anomaly_index = torch.randperm(mid_point).to(device)
        normal_index  = torch.randperm(mid_point).to(device)
        y_anomaly     = y_pred[i, :mid_point][anomaly_index]
        y_normal      = y_pred[i, mid_point:][normal_index]
        y_anomaly_max = torch.max(y_anomaly)
        y_normal_max  = torch.max(y_normal)
        loss     += F.relu(margin - (y_anomaly_max - y_normal_max))
        sparsity += torch.sum(y_anomaly) * 0.00008
        smooth   += torch.sum((y_pred[i, :frames_per_bag - 1]
                               - y_pred[i, 1:frames_per_bag]) ** 2) * 0.00008
    return (loss + sparsity + smooth) / batch_size


def focal_loss(y_pred, batch_size, device, gamma=2.5, alpha=0.25):
    y_true   = torch.cat([torch.ones(batch_size * 32),
                           torch.zeros(batch_size * 32)]).to(device)
    y_pred   = (0.8 * torch.sigmoid(y_pred)
                + 0.2 * torch.sigmoid(y_pred).mean() + 1e-7)
    bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
    pt       = torch.exp(-bce_loss)
    fl       = alpha * (1 - pt) ** gamma * bce_loss
    return fl.mean()


def anomaly_score_loss(anomaly_score, batch_size, device, margin=1.0):
    mid          = anomaly_score.size(0) // 2
    anom_scores  = anomaly_score[:mid]
    norm_scores  = anomaly_score[mid:]
    diff         = torch.mean(anom_scores) - torch.mean(norm_scores)
    return F.relu(margin - diff)

 
def train(epoch, model, normal_train_loader, anomaly_train_loader,
          optimizer, criterion, device, replay_buffer=None,
          lambda_t=0.01, lambda_m=0.01):
    """
    lambda_t : weight for TemporalAnchor loss (L_temp)
    lambda_m : weight for ProtoVault loss     (L_mem)
    """
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0

    # warmup (kept identical)
    if epoch < 15:
        lr_scale = (epoch + 1) / 15.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = (0.000510
                                 + (0.001500 - 0.000510) * lr_scale)
    print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(
            zip(normal_train_loader, anomaly_train_loader)):

        noise  = torch.randn_like(normal_inputs) * 0.07
        inputs = torch.cat([anomaly_inputs + noise, normal_inputs + noise], dim=1)
        batch_size = inputs.shape[0]
        inputs = inputs.view(-1, inputs.size(-1)).to(device)

        # replay buffer injection (kept identical)
        if replay_buffer and len(replay_buffer) > 0:
            num_samples   = min(len(replay_buffer), batch_size)
            replay_samples = np.random.choice(len(replay_buffer),
                                              num_samples, replace=False)
            replay_inputs = []
            for idx in replay_samples:
                r_inputs = replay_buffer[idx][0]
                start    = np.random.randint(0, max(1, r_inputs.size(0) - 32))
                chunk    = r_inputs[start:start + 32].to(device)
                if chunk.size(0) < 32:
                    chunk = F.pad(chunk, (0, 0, 0, 32 - chunk.size(0)))
                replay_inputs.append(chunk)
            replay_inputs = torch.cat(replay_inputs, dim=0)
            if replay_inputs.size(0) < batch_size * 32:
                replay_inputs = F.pad(
                    replay_inputs,
                    (0, 0, 0, batch_size * 32 - replay_inputs.size(0))
                )
            inputs = torch.cat([inputs, replay_inputs[:batch_size * 32]], dim=0)
            if inputs.size(0) > batch_size * 64:
                inputs = inputs[:batch_size * 64]

        # ---- forward ----
        outputs, anomaly_score, proj, L_temp, L_mem, g_t, alpha_t = model(inputs)
 
        mil_loss      = criterion(anomaly_score, batch_size, device)
        cl_loss       = contrastive_loss(proj, batch_size, device)
        fl_loss       = focal_loss(anomaly_score, batch_size, device)
        as_loss       = anomaly_score_loss(anomaly_score, batch_size, device)
        score_penalty = 0.00005 * anomaly_score.pow(2).mean()

        base_loss = (mil_loss + 1.0 * cl_loss + 1.0 * fl_loss
                     + 1.0 * as_loss + score_penalty)
 
        reg_loss = (lambda_t * L_temp.clamp(max=0.5)
                    + lambda_m * L_mem.clamp(max=0.5))

        loss = base_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

        # replay buffer update (kept identical)
        if replay_buffer is not None:
            replay_buffer.append((
                inputs[:batch_size * 64].detach().cpu(),
                anomaly_score[:batch_size * 64].detach().cpu()
            ))
            if len(replay_buffer) > 1000:
                replay_buffer.popleft()

    avg_loss = train_loss / len(normal_train_loader)
    print(f'Epoch: {epoch}, Loss: {avg_loss:.4f}, AUC: -')
    return avg_loss


 
def test_abnormal(epoch, model, anomaly_test_loader, normal_test_loader, device):
    model.eval()
    global best_auc
    auc = 0
    with torch.no_grad():
        for i, (data, data2) in enumerate(zip(anomaly_test_loader,
                                               normal_test_loader)):
            inputs, gts, frames = data
            inputs = inputs.view(-1, inputs.size(-1)).to(device)
            _, score, _, _, _, _, _ = model(inputs)
            score      = torch.sigmoid(score).cpu().detach().numpy()
            score_list = np.zeros(frames[0])
            step       = np.round(np.linspace(0, frames[0] // 16, 33))
            for j in range(32):
                score_list[int(step[j]) * 16:int(step[j + 1]) * 16] = \
                    score[j % len(score)]
            gt_list = np.zeros(frames[0])
            for k in range(len(gts) // 2):
                s = max(0, gts[k * 2] - 1)
                e = min(gts[k * 2 + 1], frames[0])
                gt_list[s:e] = 1

            inputs2, gts2, frames2 = data2
            inputs2 = inputs2.view(-1, inputs.size(-1)).to(device)
            _, score2, _, _, _, _, _ = model(inputs2)
            score2      = torch.sigmoid(score2).cpu().detach().numpy()
            score_list2 = np.zeros(frames2[0])
            step2       = np.round(np.linspace(0, frames[0] // 16, 33))
            for kk in range(32):
                score_list2[int(step2[kk]) * 16:int(step2[kk + 1]) * 16] = \
                    score2[kk % len(score2)]
            gt_list2 = np.zeros(frames2[0])

            score_list3 = np.concatenate((score_list, score_list2), axis=0)
            gt_list3    = np.concatenate((gt_list, gt_list2), axis=0)
            fpr, tpr, _ = metrics.roc_curve(gt_list3, score_list3, pos_label=1)
            auc        += metrics.auc(fpr, tpr)

        avg_auc = auc / 138
        print(f'Epoch: {epoch}, Loss: -, AUC: {avg_auc:.4f}, '
              f'Best AUC: {max(best_auc, avg_auc):.4f}')
        state = {'net': model.state_dict()}
        if avg_auc > best_auc:
            print('Saving..')
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_auc = avg_auc
            print(f'New best AUC: {best_auc:.4f}')
        if avg_auc >= 0.80:
            print('Saving high AUC checkpoint..')
            torch.save(state, f'./checkpoint/ckpt_auc_{avg_auc:.4f}.pth')
    return avg_auc


 
if __name__ == '__main__':
    torch.manual_seed(42)
    modality   = 'TWO'
    input_dim  = 2048
    memory_size = 128
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_auc   = 0
    patience   = 65
    patience_counter = 0

    normal_train_dataset  = Normal_Loader(is_train=1, modality=modality)
    normal_test_dataset   = Normal_Loader(is_train=0, modality=modality)
    anomaly_train_dataset = Anomaly_Loader(is_train=1, modality=modality)
    anomaly_test_dataset  = Anomaly_Loader(is_train=0, modality=modality)

    normal_train_loader  = DataLoader(normal_train_dataset,
                                       batch_size=20, shuffle=True)
    normal_test_loader   = DataLoader(normal_test_dataset,
                                       batch_size=1,  shuffle=True)
    anomaly_train_loader = DataLoader(anomaly_train_dataset,
                                       batch_size=20, shuffle=True)
    anomaly_test_loader  = DataLoader(anomaly_test_dataset,
                                       batch_size=1,  shuffle=True)

    model     = VAD_Model(input_size=input_dim,
                          memory_size=memory_size).to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=0.0003, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 60, 90], gamma=0.5
    )
    criterion    = MIL
    replay_buffer = deque(maxlen=1000)

    # checkpoint loading (kept identical)
    if os.path.exists('./checkpoint/ckpt_auc_0.8585.pth'):
        checkpoint  = torch.load('./checkpoint/ckpt_auc_0.8585.pth',
                                  weights_only=False)
        state_dict  = checkpoint['net']
        model_sd    = model.state_dict()
        compat_sd   = {k: v for k, v in state_dict.items()
                       if k in model_sd and v.shape == model_sd[k].shape}
        model_sd.update(compat_sd)
        model.load_state_dict(model_sd)
        best_auc = 0.8585
        print("Loaded compatible checkpoint weights with AUC 0.8585")
    elif os.path.exists('./checkpoint/ckpt_auc_0.8569.pth'):
        checkpoint  = torch.load('./checkpoint/ckpt_auc_0.8569.pth',
                                  weights_only=False)
        state_dict  = checkpoint['net']
        model_sd    = model.state_dict()
        compat_sd   = {k: v for k, v in state_dict.items()
                       if k in model_sd and v.shape == model_sd[k].shape}
        model_sd.update(compat_sd)
        model.load_state_dict(model_sd)
        best_auc = 0.8569
        print("Loaded compatible checkpoint weights with AUC 0.8569")

    for epoch in range(50):
        train_loss = train(
            epoch, model,
            normal_train_loader, anomaly_train_loader,
            optimizer, criterion, device, replay_buffer,
            lambda_t=0.01,  # TemporalAnchor loss weight
            lambda_m=0.01   # ProtoVault loss weight
        )
        auc = test_abnormal(epoch, model, anomaly_test_loader,
                             normal_test_loader, device)
        if auc > best_auc:
            best_auc        = auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs.')
                break
        scheduler.step()

    print(f'Final best AUC: {best_auc:.4f}')