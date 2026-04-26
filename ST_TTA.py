import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.init as torch_init
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import roc_curve, auc
import argparse
import random
import torch.nn.functional as F

# ── Argument parsing — KEPT IDENTICAL ─────────────────────────────────────────
parser = argparse.ArgumentParser(description='PyTorch Training with DualMemoryNetwork and MambaBlock')
parser.add_argument('--lr', default=0.0003, type=float)
parser.add_argument('--drop', default=0.35, type=float)
parser.add_argument('--feature_path', default=r"C:\Users\khanm\Desktop\lab_project\DMN\data\features\\", type=str)
parser.add_argument('--dataset', default='shanghai', type=str)
parser.add_argument('--feature_size', default=1024, type=int)
parser.add_argument('--batch_size', default=32, type=float)
parser.add_argument('--max_epoch', default=300, type=int)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── weight_init — KEPT IDENTICAL ──────────────────────────────────────────────
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


# ── Dataset — KEPT IDENTICAL ──────────────────────────────────────────────────
class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.is_normal = is_normal
        self.dataset   = args.dataset
        if self.dataset == 'shanghai':
            if test_mode:
                self.rgb_list_file = os.path.join(r"C:\Users\khanm\Desktop\lab_project\DMN\data", "shanghai-i3d-test-10crop.list")
            else:
                self.rgb_list_file = os.path.join(r"C:\Users\khanm\Desktop\lab_project\DMN\data", "shanghai-i3d-train-10crop.list")
        self.test_mode    = test_mode
        self._parse_list()
        self.feature_path = args.feature_path
        self.batch_size   = args.batch_size
        if self.test_mode:
            self.batch_size = 1
        self.pool_type = 'mean'

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        self.list = [line.strip().replace('_i3d.npy', '.npy') for line in self.list if line.strip()]
        if self.test_mode is False:
            if self.dataset == 'shanghai':
                if self.is_normal:
                    self.list = self.list[63:]
                else:
                    self.list = self.list[:63]
        if not self.list:
            raise ValueError(f"No files found in {self.rgb_list_file}")

    def get_label(self):
        if self.is_normal:
            return torch.tensor(0.0)
        return torch.tensor(1.0)

    def __getitem__(self, index):
        label = self.get_label()
        try:
            features = np.load(os.path.join(self.feature_path, self.list[index]), allow_pickle=True)
            features = np.array(features, dtype=np.float32)
            if np.all(features == 0):
                print(f"Warning: Zero features loaded for {self.list[index]}")
        except Exception as e:
            print(f"Error loading {self.list[index]}: {e}")
            features = np.zeros((32, 10, 1024), dtype=np.float32)
        segment_size = 32
        if self.test_mode:
            return features
        else:
            features = features.transpose(1, 0, 2)
            divided_features = []
            for feature in features:
                feature = process_feat(feature, segment_size)
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)
            return divided_features, label

    def __len__(self):
        return len(self.list)


 
def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
    r = np.linspace(0, len(feat), length + 1, dtype=np.int_)
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)
        else:
            new_feat[i, :] = feat[r[i], :]
    return new_feat

 
class DualMemoryNetwork(nn.Module):
    def __init__(self, input_size, memory_size=256):
        super(DualMemoryNetwork, self).__init__()
        self.primary_memory   = nn.Parameter(torch.randn(memory_size, input_size) * 0.01)
        self.secondary_memory = nn.Parameter(torch.randn(memory_size, input_size) * 0.01)

    def forward(self, feature_vector):
        batch_size = feature_vector.size(0)
        primary_memory_expanded   = self.primary_memory.unsqueeze(0).expand(batch_size, -1, -1)
        secondary_memory_expanded = self.secondary_memory.unsqueeze(0).expand(batch_size, -1, -1)
        primary_similarity   = torch.norm(primary_memory_expanded   - feature_vector.unsqueeze(1), dim=2)
        secondary_similarity = torch.norm(secondary_memory_expanded - feature_vector.unsqueeze(1), dim=2)
        primary_similarity   = primary_similarity   / (primary_similarity.max(dim=1,   keepdim=True)[0] + 1e-6)
        secondary_similarity = secondary_similarity / (secondary_similarity.max(dim=1, keepdim=True)[0] + 1e-6)
        min_similarity  = torch.min(primary_similarity, secondary_similarity)
        anomaly_score   = torch.min(min_similarity, dim=1)[0]
        return primary_similarity, secondary_similarity, anomaly_score

 
class MambaBlock(nn.Module):
    def __init__(self, input_size, hidden_size, d_state=32, d_conv=4, dropout=0.35):
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
        self.conv        = nn.Conv1d(hidden_size, hidden_size, kernel_size=d_conv,
                                     padding=d_conv - 1, groups=hidden_size)
        self.dropout     = nn.Dropout(dropout)
        self.act         = nn.SiLU()

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        residual = x
        x        = self.in_proj(x)
        x, gate  = x.chunk(2, dim=-1)
        x_conv   = x.transpose(1, 2)
        x_conv   = self.conv(x_conv)[..., :seq_len]
        x_conv   = x_conv.transpose(1, 2)
        x        = self.act(x_conv)
        x_ssm    = self.ssm_proj(x)
        h        = torch.zeros(batch_size, self.d_state, device=x.device)
        ssm_out  = []
        for t in range(seq_len):
            h   = torch.einsum('ij,bj->bi', self.A, h) + (x_ssm[:, t, :] * self.B)
            y_t = (torch.einsum('j,bj->b', self.C, h).unsqueeze(-1) * x[:, t, :]
                   + self.D * x[:, t, :])
            ssm_out.append(y_t)
        ssm_out = torch.stack(ssm_out, dim=1)
        y       = self.dropout(ssm_out * self.act(gate))
        y       = self.out_proj(y)
        y       = y + residual
        return y


# =============================================================================
# C1 — AnomalyGate
# 1a: normalised score  s_t = raw / sigma^2
# 1b: hard gate         g_t = 1[s_t < tau]
# 1c: soft modulator    alpha_t = eta * exp(-lambda * s_t)
# =============================================================================
class AnomalyGate(nn.Module):
    def __init__(self, tau=0.5, lam=2.0, eta=1.0, warmup_frames=64):
        super(AnomalyGate, self).__init__()
        self.tau           = tau
        self.lam           = lam
        self.eta           = eta
        self.warmup_frames = warmup_frames
        self.register_buffer('running_var', torch.tensor(1.0))
        self.register_buffer('frame_count', torch.tensor(0))

    @torch.no_grad()
    def _update_stats(self, raw):
        if self.frame_count < self.warmup_frames:
            n = self.frame_count.float() + 1.0
            self.running_var  = self.running_var + (raw.var() - self.running_var) / n
            self.frame_count  = self.frame_count + raw.size(0)

    def forward(self, raw_score):
        self._update_stats(raw_score.detach())
        s_t     = raw_score / self.running_var.clamp(min=1e-6)
        g_t     = (s_t < self.tau).float()
        alpha_t = self.eta * torch.exp(-self.lam * s_t)
        return s_t, g_t, alpha_t


# =============================================================================
# C2 — TemporalAnchor
# 2a: L_fp = ||p_t - p_{t-1}||^2           (frame-pair, short-range)
# 2b: L_cs = Var({p_t})                     (clip smoothness, medium-range)
# 2c: L_dr = ||z_t - z_bar_hist||^2        (drift penalty, long-range)
# =============================================================================
class TemporalAnchor(nn.Module):
    def __init__(self, rho=0.99, alpha=1.0, beta=0.5, gamma=0.5):
        super(TemporalAnchor, self).__init__()
        self.rho        = rho
        self.alpha      = alpha
        self.beta       = beta
        self.gamma      = gamma
        self.z_bar_hist = None
        self._prev_pred = None

    @torch.no_grad()
    def update_history(self, z_t, g_t):
        normal_mask = g_t.bool()
        if normal_mask.any():
            z_n = z_t[normal_mask].detach().mean(dim=0)
            self.z_bar_hist = (z_n if self.z_bar_hist is None
                               else self.rho * self.z_bar_hist + (1 - self.rho) * z_n)

    def forward(self, p_t, z_t):
        dev  = p_t.device
        loss = torch.tensor(0.0, device=dev)

        # 2a — frame-pair consistency
        if self._prev_pred is not None:
            mn   = min(p_t.size(0), self._prev_pred.size(0))
            loss = loss + self.alpha * F.mse_loss(p_t[:mn], self._prev_pred[:mn].detach().to(dev))

        # 2b — clip-level smoothness
        if p_t.size(0) > 1:
            loss = loss + self.beta * p_t.var()

        # 2c — long-range drift penalty
        if self.z_bar_hist is not None:
            z_bar = self.z_bar_hist.detach().to(dev)
            loss  = loss + self.gamma * F.mse_loss(z_t, z_bar.unsqueeze(0).expand_as(z_t))

        self._prev_pred = p_t.detach()
        return loss


# =============================================================================
# C3 — ProtoVault
# 3a: momentum update   m_k <- mu*m_k + (1-mu)*z_t      (g_t=1 only)
# 3b: pull loss         L_pull = ||z_t - m_bar_n||^2
# 3c: push loss         L_push = max(0, delta - ||z_t - m_a||)
# =============================================================================
class ProtoVault(nn.Module):
    def __init__(self, feat_dim=1024, K=64, mu=0.9, rho_a=0.99,
                 delta=1.0, alpha_p=1.0, alpha_r=0.5):
        super(ProtoVault, self).__init__()
        self.K       = K
        self.mu      = mu
        self.rho_a   = rho_a
        self.delta   = delta
        self.alpha_p = alpha_p
        self.alpha_r = alpha_r
        self.topk    = 3
        self.register_buffer('prototypes', torch.randn(K, feat_dim) * 0.01)
        self.register_buffer('m_a',        torch.zeros(feat_dim))
        self.register_buffer('m_a_init',   torch.tensor(False))

    @torch.no_grad()
    def _update_prototypes(self, z_t, g_t):
        for z in z_t[g_t.bool()].detach():
            k = torch.norm(self.prototypes - z.unsqueeze(0), dim=1).argmin()
            self.prototypes[k] = self.mu * self.prototypes[k] + (1 - self.mu) * z

    @torch.no_grad()
    def _update_anomaly_centroid(self, z_t, g_t):
        anom = z_t[~g_t.bool()].detach()
        if anom.size(0) == 0:
            return
        z_a = anom.mean(dim=0)
        if not self.m_a_init.item():
            self.m_a      = z_a
            self.m_a_init = torch.tensor(True, device=z_a.device)
        else:
            self.m_a = self.rho_a * self.m_a + (1 - self.rho_a) * z_a

    def forward(self, z_t, g_t):
        self._update_prototypes(z_t, g_t)
        self._update_anomaly_centroid(z_t, g_t)

        # 3b — prototype pull (L2, numerically stable)
        dists    = torch.cdist(z_t, self.prototypes)
        topk_idx = dists.topk(self.topk, dim=1, largest=False).indices
        m_bar    = self.prototypes[topk_idx].mean(dim=1)
        L_pull   = F.mse_loss(z_t, m_bar.detach())

        # 3c — anomaly repulsion
        if self.m_a_init.item():
            dist_a = torch.norm(z_t - self.m_a.detach().unsqueeze(0).expand_as(z_t), dim=1)
            L_push = F.relu(self.delta - dist_a).mean()
        else:
            L_push = torch.tensor(0.0, device=z_t.device)

        return self.alpha_p * L_pull + self.alpha_r * L_push

 
class Aggregate(nn.Module):
    def __init__(self, len_feature):
        super(Aggregate, self).__init__()
        self.len_feature = len_feature
        # KEPT IDENTICAL
        self.conv_1 = nn.Sequential(nn.Conv1d(len_feature, 512, kernel_size=3, stride=1, dilation=1, padding=1), nn.ReLU(), nn.BatchNorm1d(512))
        self.conv_2 = nn.Sequential(nn.Conv1d(len_feature, 512, kernel_size=3, stride=1, dilation=2, padding=2), nn.ReLU(), nn.BatchNorm1d(512))
        self.conv_3 = nn.Sequential(nn.Conv1d(len_feature, 512, kernel_size=3, stride=1, dilation=4, padding=4), nn.ReLU(), nn.BatchNorm1d(512))
        self.conv_4 = nn.Sequential(nn.Conv1d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False), nn.ReLU())
        self.conv_5 = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(), nn.BatchNorm1d(1024))
        self.mamba1         = MambaBlock(input_size=len_feature, hidden_size=len_feature, d_state=32, d_conv=4, dropout=0.35)
        self.mamba2         = MambaBlock(input_size=len_feature, hidden_size=len_feature, d_state=32, d_conv=4, dropout=0.35)
        self.memory_network = DualMemoryNetwork(input_size=len_feature, memory_size=256)
        self.recon_proj     = nn.Linear(len_feature, len_feature)
        # NEW
        self.anomaly_gate    = AnomalyGate(tau=0.5, lam=2.0, eta=1.0, warmup_frames=64)
        self.temporal_anchor = TemporalAnchor(rho=0.99, alpha=1.0, beta=0.5, gamma=0.5)
        self.proto_vault     = ProtoVault(feat_dim=len_feature, K=64, mu=0.9,
                                          rho_a=0.99, delta=1.0, alpha_p=1.0, alpha_r=0.5)

    def forward(self, x):
        # KEPT IDENTICAL — Mamba + conv pathway
        x_mamba  = self.mamba1(x)
        x_mamba  = self.mamba2(x_mamba)
        out      = x_mamba.permute(0, 2, 1)
        residual = out
        out1     = self.conv_1(out)
        out_d    = out1
        out      = self.conv_4(out)
        out      = torch.cat((out_d, out), dim=1)
        out      = self.conv_5(out)
        out      = out + residual
        out      = out.permute(0, 2, 1)

         
        final_feature = out[:, -1, :]
        _, _, anomaly_score = self.memory_network(final_feature)

        
        recon = self.recon_proj(out)
 
        s_t, g_t, alpha_t = self.anomaly_gate(anomaly_score.detach())

        
        self.temporal_anchor.update_history(final_feature, g_t)
        L_temp = self.temporal_anchor(anomaly_score, final_feature)

         
        L_mem = self.proto_vault(final_feature, g_t)

        return out, anomaly_score, recon, L_temp, L_mem

 
class Model(nn.Module):
    def __init__(self, n_features, batch_size):
        super(Model, self).__init__()
        self.batch_size   = batch_size
        self.num_segments = 32
        self.k_abn        = 1
        self.k_nor        = 1
        self.ncrops       = 10
        self.Aggregate    = Aggregate(n_features)
        self.fc1          = nn.Linear(n_features, 512)
        self.fc2          = nn.Linear(512, 128)
        self.fc3          = nn.Linear(128, 1)
        self.drop_out     = nn.Dropout(args.drop)
        self.relu         = nn.ReLU()
        self.sigmoid      = nn.Sigmoid()
        self.apply(weight_init)

    def forward(self, inputs):
        k_abn, k_nor    = self.k_abn, self.k_nor
        out             = inputs
        bs, ncrops, t, f = out.size()
        out             = out.view(-1, t, f)

        # unpack 5 (was 3)
        out, anomaly_score, recon, L_temp, L_mem = self.Aggregate(out)

        out      = self.drop_out(out)
        features = out
        scores   = self.relu(self.fc1(features))
        scores   = self.drop_out(scores)
        scores   = self.relu(self.fc2(scores))
        scores   = self.drop_out(scores)
        scores   = self.sigmoid(self.fc3(scores))
        scores   = scores.view(bs, ncrops, -1).mean(1).unsqueeze(2)

        n_size            = max(bs // 2, 1)
        normal_scores     = scores[:n_size]
        abnormal_scores   = scores[n_size:] if n_size < bs else normal_scores
        normal_features   = features[:n_size * ncrops]
        abnormal_features = features[n_size * ncrops:] if n_size < bs else normal_features
        feat_magnitudes   = torch.norm(features, p=2, dim=2).view(bs, ncrops, -1).mean(1)
        nfea_magnitudes   = feat_magnitudes[:n_size]
        afea_magnitudes   = feat_magnitudes[n_size:] if n_size < bs else nfea_magnitudes

        if nfea_magnitudes.numel() == 0:
            nfea_magnitudes = torch.zeros_like(afea_magnitudes[:1]) if afea_magnitudes.numel() > 0 else torch.zeros(1, device=device)
        if afea_magnitudes.numel() == 0:
            afea_magnitudes = torch.zeros_like(nfea_magnitudes[:1]) if nfea_magnitudes.numel() > 0 else torch.zeros(1, device=device)

        select_idx           = self.drop_out(torch.ones_like(nfea_magnitudes).to(device))
        afea_magnitudes_drop = afea_magnitudes * select_idx
        idx_abn              = torch.topk(afea_magnitudes_drop, k_abn, dim=1, largest=True)[1] if afea_magnitudes.numel() > 0 else torch.zeros((n_size, k_abn), device=device).long()
        idx_abn              = idx_abn.clamp(max=ncrops - 1)
        idx_abn_score        = idx_abn.unsqueeze(2).expand(-1, -1, 1) if idx_abn.numel() > 0 else torch.zeros((n_size, k_abn, 1), device=device).long()

        abnormal_features    = abnormal_features.view(n_size, ncrops, t, f) if abnormal_features.numel() > 0 else torch.zeros((n_size, ncrops, t, f), device=device)
        idx_abn_feat         = idx_abn.unsqueeze(2).expand(-1, -1, t).unsqueeze(-1).expand(-1, -1, -1, f)
        total_select_abn_feature = torch.gather(abnormal_features, 1, idx_abn_feat).squeeze(1)
        score_abnormal       = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score), dim=1) if abnormal_scores.numel() > 0 else torch.zeros(n_size, device=device)

        select_idx_normal    = self.drop_out(torch.ones_like(nfea_magnitudes).to(device))
        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
        idx_normal           = torch.topk(nfea_magnitudes_drop, k_nor, dim=1, largest=True)[1] if nfea_magnitudes.numel() > 0 else torch.zeros((n_size, k_nor), device=device).long()
        idx_normal           = idx_normal.clamp(max=ncrops - 1)
        idx_normal_score     = idx_normal.unsqueeze(2).expand(-1, -1, 1) if idx_normal.numel() > 0 else torch.zeros((n_size, k_nor, 1), device=device).long()

        normal_features      = normal_features.view(n_size, ncrops, t, f) if normal_features.numel() > 0 else torch.zeros((n_size, ncrops, t, f), device=device)
        idx_normal_feat      = idx_normal.unsqueeze(2).expand(-1, -1, t).unsqueeze(-1).expand(-1, -1, -1, f)
        total_select_nor_feature = torch.gather(normal_features, 1, idx_normal_feat).squeeze(1)
        score_normal         = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1) if normal_scores.numel() > 0 else torch.zeros(n_size, device=device)

        return (score_abnormal, score_normal, total_select_nor_feature,
                total_select_abn_feature, scores, feat_magnitudes,
                anomaly_score, recon, L_temp, L_mem)


 
class CombinedLoss(nn.Module):
    def __init__(self, margin=20.0, alpha=0.8, beta=0.005, gamma=0.6):
        super(CombinedLoss, self).__init__()
        self.margin = margin
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self.bce    = nn.BCELoss()

    def forward(self, score_normal, score_abnormal, nlabel, alabel,
                feat_n, feat_a, anomaly_score, recon, input):
        label = (torch.cat((nlabel, alabel), 0).to(score_normal.device)
                 if nlabel.numel() > 0 and alabel.numel() > 0
                 else torch.zeros(max(nlabel.size(0), alabel.size(0), 1), device=score_normal.device))
        if score_normal.numel() == 0 or score_abnormal.numel() == 0:
            print("Warning: Empty scores in CombinedLoss")
            return torch.tensor(0.0, device=score_normal.device, requires_grad=True), 0.0, 0.0, 0.0, 0.0
        if torch.isnan(anomaly_score).any() or torch.isinf(anomaly_score).any():
            print("Warning: Invalid anomaly scores (NaN or Inf)")
            anomaly_score = torch.clamp(anomaly_score, min=0.0, max=1.0)

        loss_bce         = self.bce(torch.cat((score_normal, score_abnormal), 0).squeeze(), label)
        feat_n_flat      = feat_n.view(feat_n.size(0), -1) if feat_n.numel() > 0 else torch.zeros(1, feat_n.size(-1), device=feat_n.device)
        feat_a_flat      = feat_a.view(feat_a.size(0), -1) if feat_a.numel() > 0 else torch.zeros(1, feat_a.size(-1), device=feat_a.device)
        min_size         = min(feat_n_flat.size(0), feat_a_flat.size(0), label.size(0))
        feat_n_flat      = feat_n_flat[:min_size]
        feat_a_flat      = feat_a_flat[:min_size]
        dist             = (F.pairwise_distance(feat_n_flat, feat_a_flat)
                            if feat_n_flat.numel() > 0 and feat_a_flat.numel() > 0
                            else torch.tensor(0.0, device=feat_n.device))
        loss_contrastive = (torch.mean(
            (1 - label[:dist.size(0)]) * torch.pow(dist, 2) +
            label[:dist.size(0)] * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
            if dist.numel() > 0 else torch.tensor(0.0, device=dist.device))
        loss_contrastive = torch.log1p(loss_contrastive)
        anomaly_diff     = (torch.mean(anomaly_score[nlabel.size(0):]) - torch.mean(anomaly_score[:nlabel.size(0)])
                            if nlabel.size(0) < anomaly_score.size(0)
                            else torch.tensor(0.0, device=anomaly_score.device))
        loss_anomaly     = self.gamma * torch.clamp(-anomaly_diff + 1.0, min=0.0)
        loss_recon       = (self.beta * nn.MSELoss()(recon, input.view(-1, 32, 1024))
                            if recon.numel() > 0 and input.numel() > 0
                            else torch.tensor(0.0, device=input.device))
        total_loss = loss_bce + self.alpha * loss_contrastive + loss_anomaly + loss_recon
        return total_loss, loss_bce.item(), loss_contrastive.item(), loss_anomaly.item(), loss_recon.item()


 
class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.capacity        = capacity
        self.normal_buffer   = []
        self.abnormal_buffer = []

    def add(self, input, label, is_normal):
        sample = (input.cpu(), label.cpu())
        buffer = self.normal_buffer if is_normal else self.abnormal_buffer
        buffer.append(sample)
        if len(buffer) > self.capacity:
            buffer.pop(0)

    def sample(self, n_samples, device):
        normal_samples   = random.sample(self.normal_buffer,   min(n_samples, len(self.normal_buffer)))   if self.normal_buffer   else []
        abnormal_samples = random.sample(self.abnormal_buffer, min(n_samples, len(self.abnormal_buffer))) if self.abnormal_buffer else []
        inputs, labels   = [], []
        for inp, lbl in normal_samples + abnormal_samples:
            inputs.append(inp.to(device))
            labels.append(lbl.to(device))
        if inputs:
            inputs = torch.cat(inputs, dim=0)
            labels = torch.cat(labels, dim=0)
        else:
            inputs = torch.empty(0, 10, 32, 1024, device=device)
            labels = torch.empty(0, device=device)
        return inputs, labels


 
def train(nloader, aloader, model, batch_size, optimizer, scheduler,
          warmup_epochs, device, epoch, accumulation_steps=2,
          replay_buffer=None, lambda_t=0.01, lambda_m=0.01):
    model.train()
    optimizer.zero_grad()
    total_loss = 0
    for i in range(accumulation_steps):
        try:
            ninput, nlabel = next(nloader)
        except StopIteration:
            nloader = iter(train_nloader)
            ninput, nlabel = next(nloader)
        try:
            ainput, alabel = next(aloader)
        except StopIteration:
            aloader = iter(train_aloader)
            ainput, alabel = next(aloader)

        if ninput.size(0) != batch_size // accumulation_steps or ainput.size(0) != batch_size // accumulation_steps:
            print(f"Warning: Batch size mismatch - normal: {ninput.size(0)}, abnormal: {ainput.size(0)}, expected: {batch_size // accumulation_steps}")
            continue

        # KEPT IDENTICAL — replay buffer logic
        for ni, nl in zip(ninput, nlabel):
            replay_buffer.add(ni.unsqueeze(0), nl.unsqueeze(0), is_normal=True)
        for ai, al in zip(ainput, alabel):
            replay_buffer.add(ai.unsqueeze(0), al.unsqueeze(0), is_normal=False)
        replay_inputs, replay_labels = replay_buffer.sample(n_samples=12, device=device)

        nlabel        = nlabel.to(device)
        alabel        = alabel.to(device)
        current_input = torch.cat((ninput, ainput), 0).to(device)
        current_label = torch.cat((nlabel, alabel), 0)
        if replay_inputs.numel() > 0:
            input = torch.cat((current_input, replay_inputs), 0)
            label = torch.cat((current_label, replay_labels), 0)
        else:
            input = current_input
            label = current_label

        # unpack 10 values (was 8)
        (score_abnormal, score_normal, feat_select_nor, feat_select_abn,
         scores, _, anomaly_score, recon, L_temp, L_mem) = model(input)

        scores     = scores.view(input.size(0) * 32, -1).squeeze()
        abn_scores = scores[(input.size(0) // 2) * 32:] if input.size(0) > 0 else scores
        nlabel     = label[:input.size(0) // 2] if input.size(0) > 0 else label
        alabel     = label[input.size(0) // 2:] if input.size(0) > 0 else label

        loss_criterion = CombinedLoss(margin=20.0, alpha=0.8, beta=0.005, gamma=0.6)
        (total_loss_step, loss_bce, loss_contrastive,
         loss_anomaly, loss_recon) = loss_criterion(
            score_normal, score_abnormal, nlabel, alabel,
            feat_select_nor, feat_select_abn, anomaly_score, recon, input)

        # add C2 + C3 as small clamped regularizers — never dominate base loss
        reg_loss        = lambda_t * L_temp.clamp(max=0.5) + lambda_m * L_mem.clamp(max=0.5)
        total_loss_step = total_loss_step + reg_loss

        total_loss += total_loss_step.item() / accumulation_steps
        if total_loss_step == 0:
            print(f"Warning: Zero loss in accumulation step {i}")
        total_loss_step.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            if epoch < warmup_epochs or (epoch >= 100 and total_loss > 8.0):
                scheduler.step()
            optimizer.zero_grad()

    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch: {epoch}, Learning Rate: {lr:.6f}, Loss: {total_loss:.4f}, "
          f"BCE: {loss_bce:.4f}, Contrastive: {loss_contrastive:.4f}, "
          f"Anomaly: {loss_anomaly:.4f}, Recon: {loss_recon:.4f}")
    return total_loss, nloader, aloader


 
def test(dataloader, model, args, device, epoch):
    model.eval()
    global best_auc
    pred          = torch.zeros(0, device=device)
    memory_scores = torch.zeros(0, device=device)
    with torch.no_grad():
        for input in dataloader:
            input = input.to(device).permute(0, 2, 1, 3)
            # unpack 10 (was 8)
            (score_abnormal, score_normal, _, _, logits, _,
             anomaly_score, _, L_temp, L_mem) = model(input)
            logits        = torch.squeeze(logits, 2).mean(0)
            anomaly_score = anomaly_score.mean()
            pred          = torch.cat((pred, logits))
            memory_scores = torch.cat((memory_scores, anomaly_score.unsqueeze(0)))

    gt = np.load(os.path.join(r"C:\Users\khanm\Desktop\lab_project\DMN\data", "gt-sh-test.npy"))
    pred          = pred.cpu().numpy()
    memory_scores = memory_scores.cpu().numpy()
    if np.all(pred == pred[0]) or np.all(memory_scores == memory_scores[0]):
        print("Warning: Constant predictions or memory scores detected")
    memory_scores    = np.repeat(memory_scores, 32)
    target_length    = len(gt)
    orig_length_pred = len(pred)
    orig_length_mem  = len(memory_scores)
    x                = np.arange(orig_length_pred)
    x_new            = np.linspace(0, orig_length_pred - 1, target_length)
    repeated_pred    = np.interp(x_new, x, pred)
    x_mem            = np.arange(orig_length_mem)
    x_new_mem        = np.linspace(0, orig_length_mem - 1, target_length)
    repeated_memory_scores = np.interp(x_new_mem, x_mem, memory_scores)
    if len(repeated_pred) != len(gt) or len(repeated_memory_scores) != len(gt):
        raise ValueError(f"Final shapes do not match gt length {len(gt)}: "
                         f"pred {len(repeated_pred)}, memory_scores {len(repeated_memory_scores)}")
    final_scores = 0.7 * repeated_pred + 0.3 * repeated_memory_scores
    fpr, tpr, _  = roc_curve(gt, final_scores)
    avg_auc      = auc(fpr, tpr)
    print(f"Epoch: {epoch}, AUC: {avg_auc:.4f}, Best AUC: {max(best_auc, avg_auc):.4f}")
    return avg_auc


 
if __name__ == '__main__':
    torch.manual_seed(42)
    best_auc         = 0
    patience         = 200
    patience_counter = 0
    warmup_epochs    = 25

    train_nloader = DataLoader(Dataset(args, is_normal=True,  test_mode=False),
                               batch_size=args.batch_size // 2, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args, is_normal=False, test_mode=False),
                               batch_size=args.batch_size // 2, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    test_loader   = DataLoader(Dataset(args, test_mode=True),
                               batch_size=1, shuffle=False,
                               num_workers=0, pin_memory=False)

    model     = Model(args.feature_size, args.batch_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.05)

    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.998 ** (epoch - warmup_epochs + 1)

    scheduler     = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    replay_buffer = ReplayBuffer(capacity=1000)
    loadern_iter  = iter(train_nloader)
    loadera_iter  = iter(train_aloader)

    for epoch in range(args.max_epoch):
        result = train(loadern_iter, loadera_iter, model, args.batch_size,
                       optimizer, scheduler, warmup_epochs, device, epoch,
                       accumulation_steps=2, replay_buffer=replay_buffer,
                       lambda_t=0.01, lambda_m=0.01)
        if result is None:
            continue
        loss, loadern_iter, loadera_iter = result
        auc_res = test(test_loader, model, args, device, epoch)
        if auc_res > best_auc:
            best_auc         = auc_res
            patience_counter = 0
            if not os.path.isdir('checkpoint3'):
                os.mkdir('checkpoint3')
            torch.save(model.state_dict(), './checkpoint3/ckpt3.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch} epochs.')
                break
        # KEPT IDENTICAL — lr boost on plateau
        if epoch >= 100 and auc_res < 0.97:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 1.5
                print(f"Boosted learning rate to {param_group['lr']:.6f} at epoch {epoch}")

    print(f'Final best AUC: {best_auc:.4f}')