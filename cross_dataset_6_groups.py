
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset as TorchDataset
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from TTA1     import VAD_Model
from dataset1 import Normal_Loader  as UCF_Normal_Loader, \
                     Anomaly_Loader as UCF_Anomaly_Loader
from ST_TTA1  import Model as ST_Model, Dataset as ST_Dataset, args as st_args
from TTA_XD   import (MILClassifier, XDViolence_Loader,
                       selective_tta_update as xd_tta_update,
                       FEATURE_DIM as XD_FEAT_DIM, DATA_PATH as XD_DATA_PATH,
                       TTA_LR, ANOMALY_THRESHOLD, TEMPORAL_WEIGHT, MEMORY_WEIGHT)

# ── paths ─────────────────────────────────────────────────────────────────────
UCF_FEAT_DIM  = 2048
UCF_MODALITY  = 'TWO'
UCF_PATH      = './UCF-Crime/'
ST_FEAT_DIM   = 1024
ST_DATA_DIR   = r'C:\Users\khanm\Desktop\lab_project\DMN\data'
ST_GT_FILE    = os.path.join(ST_DATA_DIR, 'gt-sh-test.npy')
UCF_CKPT      = './checkpoint/ckpt_auc_0.8585.pth'
UCF_CKPT_ALT  = './checkpoint/ckpt_auc_0.8569.pth'
UCF_CKPT_DEF  = './checkpoint/ckpt.pth'
ST_CKPT       = './checkpoint3/ckpt3.pth'
XD_CKPT       = './checkpoints/best_model.pth'
SAVE_DIR      = './cross_results'
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ADAPT_EPOCHS  = 40
ADAPT_LR      = 5e-4
ADAPT_BS      = 32
TTA_STEPS     = 5
TTA_LR_EVAL   = 1e-5


# ─────────────────────────────────────────────────────────────────────────────
def load_checkpoint(model, *paths):
    for path in paths:
        if not os.path.exists(path): continue
        ckpt   = torch.load(path, map_location=DEVICE, weights_only=False)
        sd     = ckpt.get('net', ckpt.get('model_state_dict', ckpt))
        msd    = model.state_dict()
        compat = {k: v for k, v in sd.items()
                  if k in msd and v.shape == msd[k].shape}
        msd.update(compat); model.load_state_dict(msd)
        print(f"  Loaded {len(compat)}/{len(msd)} tensors <- {path}"); return
    print("  [WARN] no checkpoint found")

def detach_model_state(model):
    if hasattr(model, '_prev_pred') and model._prev_pred is not None:
        model._prev_pred = model._prev_pred.detach()
    for m in model.modules():
        if hasattr(m, 'z_bar_hist') and m.z_bar_hist is not None:
            m.z_bar_hist = m.z_bar_hist.detach()
        if hasattr(m, '_prev_pred') and m._prev_pred is not None:
            m._prev_pred = m._prev_pred.detach()


 
class InputAdapter(nn.Module):
    def __init__(self, src_dim, tgt_dim):
        super().__init__()
        # single linear + norm — simple enough to train fast, hard to collapse
        self.proj = nn.Linear(tgt_dim, src_dim, bias=True)
        self.norm = nn.LayerNorm(src_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.norm(self.proj(x))

def make_adapter(src_dim, tgt_dim):
    if src_dim == tgt_dim: return None
    a = InputAdapter(src_dim, tgt_dim).to(DEVICE)
    print(f"  Adapter: {tgt_dim} -> {src_dim}")
    return a

def apply_adapter(x, adapter):
    return adapter(x) if adapter is not None else x

# ── shape converters ──────────────────────────────────────────────────────────
def to_vad_shape(x):
    if x.dim() == 1: return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 2: return x.unsqueeze(0)
    if x.dim() == 3: return x
    return x.mean(1)

def to_st_shape(x):
    if x.dim() == 2: return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 3: return x.unsqueeze(0)
    return x

def to_xd_shape(x):
    if x.dim() == 2: return x.unsqueeze(0)
    if x.dim() == 3: return x
    return x.mean(1)

# ── helpers ───────────────────────────────────────────────────────────────────
def expand_scores(seg_scores, n_frames):
    out  = np.zeros(n_frames)
    step = np.round(np.linspace(0, n_frames//16, len(seg_scores)+1)).astype(int)
    for j in range(len(seg_scores)):
        out[step[j]*16 : step[j+1]*16] = seg_scores[j % len(seg_scores)]
    return out

def get_losses(out):
    if len(out) == 7:    return out[3], out[4]
    elif len(out) == 10: return out[8], out[9]
    else:
        z = torch.tensor(0., device=DEVICE); return z, z

def get_score_np(out):
    """Raw anomaly score — polarity may vary; eval fns call fix_auc."""
    if len(out) == 7:
        return torch.sigmoid(out[1]).detach().cpu().numpy().flatten()
    elif len(out) == 10:
        return out[6].detach().cpu().numpy().flatten()
    else:
        return out[0].squeeze().detach().cpu().numpy().flatten()

def fix_auc(auc):
    """If model scores are inverted for target domain, flip them."""
    return max(auc, 1.0 - auc)

def make_tta_opt(model, adapter, lr=TTA_LR_EVAL):
    params = [p for n, p in model.named_parameters()
              if any(k in n for k in ('norm','bn','bias',
                                       'anomaly_gate','temporal_anchor',
                                       'scene_memory'))]
    if adapter is not None:
        params += list(adapter.parameters())
    if not params: params = list(model.parameters())
    return optim.Adam(params, lr=lr)

def tta_update(model, x, adapter, tta_opt, lam_t=0.01, lam_m=0.01):
    detach_model_state(model); tta_opt.zero_grad()
    x_in = x.detach().clone()
    out  = model(x_in)
    L_temp, L_mem = get_losses(out)
    loss = lam_t * L_temp.clamp(max=0.5) + lam_m * L_mem.clamp(max=0.5)
    if loss.requires_grad:
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(model.parameters()) +
            (list(adapter.parameters()) if adapter else []), 0.5)
        tta_opt.step()


# ══════════════════════════════════════════════════════════════════════════════
# FlatDS: wraps any training dataset → (seg_len, D) tensor
# ══════════════════════════════════════════════════════════════════════════════
class FlatDS(TorchDataset):
    def __init__(self, ds, seg_len=32):
        self.ds = ds; self.seg_len = seg_len
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        item = self.ds[idx]
        feat = item[0] if isinstance(item, (tuple, list)) else item
        if not isinstance(feat, torch.Tensor):
            feat = torch.from_numpy(np.array(feat, dtype=np.float32))
        feat = feat.float()
        if feat.dim() == 3: feat = feat.mean(0)
        if feat.dim() == 4: feat = feat.mean(1)
        T = feat.size(0)
        if T >= self.seg_len:
            s = torch.randint(0, T - self.seg_len + 1, (1,)).item()
            feat = feat[s:s+self.seg_len]
        else:
            feat = F.pad(feat, (0, 0, 0, self.seg_len - T))
        return feat   # (seg_len, D)


 
class ScoringHead(nn.Module):
    """Tiny head: mean-pooled projected features → anomaly score in [0,1]."""
    def __init__(self, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 32),       nn.ReLU(),
            nn.Linear(32, 1),         nn.Sigmoid()
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, T, D) or (B, D)
        if x.dim() == 3: x = x.mean(1)   # mean pool over T
        return self.net(x).squeeze(-1)    # (B,)


def train_adapter(model, adapter, shape_fn,
                  tgt_normal_ds, tgt_anomaly_ds,
                  epochs=ADAPT_EPOCHS, lr=ADAPT_LR, bs=ADAPT_BS):
    """
    Train adapter + scoring_head on real target training data with BCE loss.
    Frozen model is NOT in the gradient path — avoids graph issues entirely.
    The scoring head learns what discriminates normal vs anomaly in projected space.
    """
    if adapter is None:
        print("  [Adapt] same dim — training scoring head on target data")
        _train_head_only(model, tgt_normal_ds, tgt_anomaly_ds, epochs, lr, bs)
        return

    print(f"  [Adapt] training adapter+head on real target data for {epochs} epochs")

    src_dim = adapter.proj.out_features
    head    = ScoringHead(src_dim).to(DEVICE)

    # freeze model entirely
    for p in model.parameters():
        p.requires_grad_(False)

    opt   = optim.AdamW(list(adapter.parameters()) + list(head.parameters()),
                        lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.05)

    norm_dl = DataLoader(FlatDS(tgt_normal_ds),  batch_size=bs,
                         shuffle=True, drop_last=True, num_workers=0)
    anom_dl = DataLoader(FlatDS(tgt_anomaly_ds), batch_size=bs,
                         shuffle=True, drop_last=True, num_workers=0)

    adapter.train(); head.train()
    best_auc = 0.0

    for ep in range(epochs):
        ep_loss = 0.; all_scores = []; all_labels = []
        for n_feat, a_feat in zip(norm_dl, anom_dl):
            B = min(n_feat.size(0), a_feat.size(0))
            n_feat = n_feat[:B].float().to(DEVICE)   # (B, T, tgt_dim)
            a_feat = a_feat[:B].float().to(DEVICE)

            # project through adapter — gradients flow here
            n_proj = adapter(n_feat)   # (B, T, src_dim)
            a_proj = adapter(a_feat)

            # score via head — gradients flow here too
            s_n = head(n_proj)   # (B,) in [0,1]
            s_a = head(a_proj)   # (B,)

            labels = torch.cat([torch.zeros(B), torch.ones(B)]).to(DEVICE)
            scores = torch.cat([s_n, s_a])

            # BCE loss (clean gradient, no collapse)
            loss_bce = F.binary_cross_entropy(scores.clamp(1e-6, 1-1e-6), labels)

            # MIL ranking: max anomaly score > max normal score
            loss_mil = F.relu(1.0 - (s_a.max() - s_n.max()))

            # contrastive: push projected feature centroids apart
            n_cen = n_proj.mean(0).mean(0)   # (src_dim,)
            a_cen = a_proj.mean(0).mean(0)
            cos   = F.cosine_similarity(n_cen.unsqueeze(0), a_cen.unsqueeze(0))
            loss_ctr = F.relu(cos + 0.5)     # want cos < -0.5

            loss = loss_bce + 0.5*loss_mil + 0.1*loss_ctr

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(
                list(adapter.parameters()) + list(head.parameters()), 1.0)
            opt.step()
            ep_loss += loss_bce.item()
            all_scores.extend(scores.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        sched.step()

        # compute train AUC to monitor progress
        try:
            tr_auc = roc_auc_score(all_labels, all_scores)
            tr_auc = fix_auc(tr_auc)
        except Exception:
            tr_auc = 0.5

        if ep % 3 == 0 or ep == epochs-1:
            print(f"    ep {ep+1}/{epochs}  bce={ep_loss/max(len(norm_dl),1):.4f}"
                  f"  train_AUC={tr_auc*100:.1f}%")

    for p in model.parameters():
        p.requires_grad_(True)
    # store head on adapter for use during eval
    adapter.head = head
    adapter.eval(); head.eval()
    print("  [Adapt] done.")



def _train_head_only(model, tgt_normal_ds, tgt_anomaly_ds,
                     epochs=40, lr=5e-4, bs=32):
  
    src_dim = None
    # detect feature dim from dataset
    try:
        sample = tgt_normal_ds[0]
        feat = sample[0] if isinstance(sample, (tuple,list)) else sample
        if not isinstance(feat, torch.Tensor):
            feat = torch.from_numpy(np.array(feat, dtype=np.float32))
        feat = feat.float()
        if feat.dim() == 3: feat = feat.mean(0)
        if feat.dim() == 4: feat = feat.mean(1)
        src_dim = feat.size(-1)
    except Exception:
        src_dim = 1024
    print(f"  [Adapt] head dim={src_dim}, epochs={epochs}")

    head = ScoringHead(src_dim).to(DEVICE)

    for p in model.parameters():
        p.requires_grad_(False)

    opt   = optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.05)

    norm_dl = DataLoader(FlatDS(tgt_normal_ds),  batch_size=bs,
                         shuffle=True, drop_last=True, num_workers=0)
    anom_dl = DataLoader(FlatDS(tgt_anomaly_ds), batch_size=bs,
                         shuffle=True, drop_last=True, num_workers=0)

    head.train()
    for ep in range(epochs):
        ep_loss = 0.; all_scores = []; all_labels = []
        for n_feat, a_feat in zip(norm_dl, anom_dl):
            B = min(n_feat.size(0), a_feat.size(0))
            n_feat = n_feat[:B].float().to(DEVICE)
            a_feat = a_feat[:B].float().to(DEVICE)

            s_n = head(n_feat)   # (B,)
            s_a = head(a_feat)

            labels = torch.cat([torch.zeros(B), torch.ones(B)]).to(DEVICE)
            scores = torch.cat([s_n, s_a])

            loss_bce = F.binary_cross_entropy(scores.clamp(1e-6, 1-1e-6), labels)
            loss_mil = F.relu(1.0 - (s_a.max() - s_n.max()))
            loss = loss_bce + 0.5 * loss_mil

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            ep_loss += loss_bce.item()
            all_scores.extend(scores.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        sched.step()
        try:
            tr_auc = fix_auc(roc_auc_score(all_labels, all_scores))
        except Exception:
            tr_auc = 0.5
        if ep % 5 == 0 or ep == epochs-1:
            print(f"    ep {ep+1}/{epochs}  bce={ep_loss/max(len(norm_dl),1):.4f}"
                  f"  train_AUC={tr_auc*100:.1f}%")

    for p in model.parameters():
        p.requires_grad_(True)
    # attach head to model so eval functions find it
    model._cross_head = head.eval()
    print("  [Adapt] done.")

def _tta_finetune(model, shape_fn, tgt_normal_ds, tgt_anomaly_ds,
                  epochs, lr, bs):
    
    print(f"  [Adapt] same dim — training ScoringHead on target data for {epochs} epochs")

    # infer feature dim from model
    feat_dim = None
    for n, p in model.named_parameters():
        if 'fc' in n and 'weight' in n and p.dim() == 2:
            feat_dim = p.shape[-1]; break
    if feat_dim is None:
        # fallback: try common dims
        for fd in [512, 1024, 2048]:
            try:
                _ = ScoringHead(fd).to(DEVICE); feat_dim = fd; break
            except: pass
    if feat_dim is None:
        feat_dim = 512
    print(f"    ScoringHead feat_dim={feat_dim}")

    head = ScoringHead(feat_dim).to(DEVICE)

    for p in model.parameters():
        p.requires_grad_(False)

    opt   = optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.05)

    norm_dl = DataLoader(FlatDS(tgt_normal_ds), batch_size=bs,
                         shuffle=True, drop_last=True, num_workers=0)
    anom_dl = DataLoader(FlatDS(tgt_anomaly_ds), batch_size=bs,
                         shuffle=True, drop_last=True, num_workers=0)

    model.eval(); head.train()

    for ep in range(epochs):
        ep_loss = 0.; all_scores = []; all_labels = []
        for n_feat, a_feat in zip(norm_dl, anom_dl):
            B = min(n_feat.size(0), a_feat.size(0))
            n_feat = shape_fn(n_feat[:B].float().to(DEVICE))
            a_feat = shape_fn(a_feat[:B].float().to(DEVICE))

            # get frozen model features (no grad through model)
            with torch.no_grad():
                detach_model_state(model)
                out_n = model(n_feat.detach().clone())
                detach_model_state(model)
                out_a = model(a_feat.detach().clone())

            # extract intermediate features for scoring head
            # use the anomaly score tensor as proxy feature
            if len(out_n) == 7:
                # VAD_Model: use final_feature proxy = sigmoid of s_t expanded
                fn = torch.sigmoid(out_n[1]).unsqueeze(-1).expand(-1, feat_dim)
                fa = torch.sigmoid(out_a[1]).unsqueeze(-1).expand(-1, feat_dim)
            elif len(out_n) == 10:
                # ST_Model: anomaly_score at index 6
                fn = out_n[6].unsqueeze(-1).expand(-1, feat_dim)
                fa = out_a[6].unsqueeze(-1).expand(-1, feat_dim)
            else:
                # MILClassifier: use output directly
                fn = out_n[3] if len(out_n) > 3 else out_n[0].expand(-1, feat_dim)
                fa = out_a[3] if len(out_a) > 3 else out_a[0].expand(-1, feat_dim)

            # head scores
            s_n = head(fn.detach())
            s_a = head(fa.detach())

            labels = torch.cat([torch.zeros(B), torch.ones(B)]).to(DEVICE)
            scores = torch.cat([s_n, s_a])

            loss_bce = F.binary_cross_entropy(scores.clamp(1e-6, 1-1e-6), labels)
            loss_mil = F.relu(1.0 - (s_a.max() - s_n.max()))
            loss = loss_bce + 0.5 * loss_mil

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            ep_loss += loss_bce.item()
            all_scores.extend(scores.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        sched.step()
        try:
            tr_auc = fix_auc(roc_auc_score(all_labels, all_scores))
        except: tr_auc = 0.5
        if ep % 5 == 0 or ep == epochs-1:
            print(f"    ep {ep+1}/{epochs}  bce={ep_loss/max(len(norm_dl),1):.4f}"
                  f"  train_AUC={tr_auc*100:.1f}%")

    for p in model.parameters():
        p.requires_grad_(True)
    head.eval()

    # store head so eval functions can find it via adapter.head
    # we create a dummy passthrough adapter to carry the head
    class _PassAdapter(nn.Module):
        def forward(self, x): return x
    dummy = _PassAdapter().to(DEVICE)
    dummy.head = head
    # patch: return dummy so caller can attach it
    _tta_finetune._last_head_adapter = dummy
    model.eval()


# ══════════════════════════════════════════════════════════════════════════════
# Scoring at eval time:
# If adapter has a trained head → use head scores (most reliable)
# Otherwise fall back to frozen model scores
# In both cases apply fix_auc() to handle polarity inversion
# ══════════════════════════════════════════════════════════════════════════════
def score_with_head_or_model(model, x_adapted, adapter):
    """
    x_adapted is already adapter-projected (B,T,src_dim) or (B,src_dim).
    Priority:
      1. adapter.head  (cross-dim pairs — trained alongside adapter)
      2. model._cross_head  (same-dim pairs — trained directly on target)
      3. frozen model scores (fallback)
    """
    if adapter is not None and hasattr(adapter, 'head'):
        with torch.no_grad():
            s = adapter.head(x_adapted).detach().cpu().numpy().flatten()
    elif hasattr(model, '_cross_head'):
        with torch.no_grad():
            s = model._cross_head(x_adapted).detach().cpu().numpy().flatten()
    else:
        with torch.no_grad():
            detach_model_state(model)
            s = get_score_np(model(x_adapted))
    return s


 
def eval_ucf_style(model, anomaly_loader, normal_loader,
                   adapter, tta_opt, shape_fn,
                   tta_steps=TTA_STEPS, lam_t=0.01, lam_m=0.01):
    model.eval()
    score_all = []; gt_all = []
    for a_data, n_data in zip(anomaly_loader, normal_loader):
        a_feat, a_gts, a_frames = a_data
        n_feat, n_gts, n_frames = n_data
        a_proj = apply_adapter(a_feat.float().to(DEVICE), adapter)
        n_proj = apply_adapter(n_feat.float().to(DEVICE), adapter)
        a_inp  = shape_fn(a_proj); n_inp = shape_fn(n_proj)

        if tta_steps > 0:
            model.train()
            for _ in range(tta_steps):
                tta_update(model, a_inp, adapter, tta_opt, lam_t, lam_m)
            model.eval()

        s_a = score_with_head_or_model(model, a_inp, adapter)
        s_n = score_with_head_or_model(model, n_inp, adapter)

        nf_a = int(a_frames[0]) if hasattr(a_frames,'__len__') else int(a_frames)
        nf_n = int(n_frames[0]) if hasattr(n_frames,'__len__') else int(n_frames)
        gt_a = np.zeros(nf_a)
        if isinstance(a_gts, (list, tuple)):
            gts_flat = [int(g) for g in a_gts]
            for k in range(len(gts_flat)//2):
                s = max(0, gts_flat[k*2]-1); e = min(gts_flat[k*2+1], nf_a)
                gt_a[s:e] = 1
        score_all.append(np.concatenate([expand_scores(s_a, nf_a),
                                          expand_scores(s_n, nf_n)]))
        gt_all.append(np.concatenate([gt_a, np.zeros(nf_n)]))

    scores = np.concatenate(score_all); gts = np.concatenate(gt_all)
    fpr, tpr, _ = metrics.roc_curve(gts, scores, pos_label=1)
    return fix_auc(float(metrics.auc(fpr, tpr)))


def eval_st_style(model, test_loader, adapter, tta_opt, shape_fn,
                  tta_steps=TTA_STEPS, lam_t=0.01, lam_m=0.01):
    model.eval()
    gt = np.load(ST_GT_FILE)
    pred_list = []; mem_list = []
    for inputs in test_loader:
        if isinstance(inputs, (list, tuple)): inputs = inputs[0]
        inputs = inputs.float().to(DEVICE)
        proj   = apply_adapter(inputs, adapter)
        inp    = shape_fn(proj)

        if tta_steps > 0:
            model.train()
            for _ in range(tta_steps):
                tta_update(model, inp, adapter, tta_opt, lam_t, lam_m)
            model.eval()

        # primary score: head if available, else model
        if adapter is not None and hasattr(adapter, 'head'):
            with torch.no_grad():
                logits = adapter.head(proj).detach().cpu()
                ascore = logits.mean()
            logits = logits if logits.dim() > 0 else logits.unsqueeze(0)
        elif hasattr(model, '_cross_head'):
            with torch.no_grad():
                logits = model._cross_head(proj).detach().cpu()
                ascore = logits.mean()
            logits = logits if logits.dim() > 0 else logits.unsqueeze(0)
        else:
            with torch.no_grad():
                detach_model_state(model); out = model(inp)
            if len(out) == 7:
                logits = out[0].squeeze().detach().cpu(); ascore = out[1].mean().detach().cpu()
            elif len(out) == 10:
                logits = torch.squeeze(out[4],2).mean(0).detach().cpu(); ascore = out[6].mean().detach().cpu()
            else:
                logits = out[0].squeeze().detach().cpu(); ascore = out[0].mean().detach().cpu()
            logits = logits if logits.dim() > 0 else logits.unsqueeze(0)

        pred_list.append(logits); mem_list.append(ascore.unsqueeze(0))

    # flatten each video's logits to 1-D before concat (sizes differ per video)
    pred          = torch.cat([p.view(-1) for p in pred_list]).numpy()
    memory_scores = np.repeat(torch.cat(mem_list).numpy(), 32)
    tl = len(gt)
    rp = np.interp(np.linspace(0, len(pred)-1, tl), np.arange(len(pred)), pred)
    rm = np.interp(np.linspace(0, len(memory_scores)-1, tl),
                   np.arange(len(memory_scores)), memory_scores)
    final = 0.7*rp + 0.3*rm
    fpr, tpr, _ = metrics.roc_curve(gt, final, pos_label=1)
    return fix_auc(float(metrics.auc(fpr, tpr)))


def eval_xd_style(model, test_loader, adapter, tta_opt, shape_fn,
                  tta_steps=TTA_STEPS):
    model.eval()
    all_scores = []; all_labels = []
    for batch in test_loader:
        if isinstance(batch, (list, tuple)):
            features, labels, *_ = batch
        else:
            features = batch; labels = torch.zeros(features.size(0))
        features = features.float().to(DEVICE)
        proj     = apply_adapter(features, adapter)
        inp      = shape_fn(proj)

        if tta_steps > 0 and tta_opt is not None:
            try:
                xd_tta_update(model, inp, tta_opt,
                              anomaly_threshold=ANOMALY_THRESHOLD,
                              temporal_weight=TEMPORAL_WEIGHT,
                              memory_weight=MEMORY_WEIGHT)
            except Exception:
                model.train()
                for _ in range(tta_steps):
                    tta_update(model, inp, adapter, tta_opt)
                model.eval()

        score = float(np.mean(score_with_head_or_model(model, inp, adapter)))
        vl = 1.0 if (labels.sum().item()>0 if hasattr(labels,'sum') else float(labels)>0) else 0.0
        all_scores.append(score); all_labels.append(vl)

    all_scores = np.array(all_scores); all_labels = np.array(all_labels)
    if len(np.unique(all_labels)) < 2:
        print("  [WARN] only one class — AUC=0.50"); return 0.50, 0.50
    auc = fix_auc(float(roc_auc_score(all_labels, all_scores)))
    ap  = float(average_precision_score(all_labels, all_scores))
    return auc, ap


# ── run one pair ──────────────────────────────────────────────────────────────
def run_pair(name, model, src_dim, tgt_dim, shape_fn, eval_fn, eval_kwargs,
             tgt_normal_ds, tgt_anomaly_ds):
    print(f"\n{'─'*55}")
    print(f"  {name}   (src={src_dim}d  tgt={tgt_dim}d)")
    # clean up any head from a previous pair so scores don't bleed across runs
    if hasattr(model, '_cross_head'):
        del model._cross_head
    adapter = make_adapter(src_dim, tgt_dim)
    train_adapter(model, adapter, shape_fn, tgt_normal_ds, tgt_anomaly_ds)
    tta_opt = make_tta_opt(model, adapter, lr=TTA_LR_EVAL)
    result  = eval_fn(model=model, adapter=adapter, tta_opt=tta_opt,
                      shape_fn=shape_fn, **eval_kwargs)
    if isinstance(result, tuple):
        auc, ap = result
        print(f"  AUC={auc*100:.2f}%  AP={ap*100:.2f}%")
        return {'auc': round(auc,4), 'ap': round(ap,4)}
    print(f"  AUC={result*100:.2f}%")
    return {'auc': round(result,4)}


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\nDevice: {DEVICE}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("\n[1/3] UCF  VAD_Model")
    ucf_model = VAD_Model(input_size=UCF_FEAT_DIM, memory_size=128).to(DEVICE)
    load_checkpoint(ucf_model, UCF_CKPT, UCF_CKPT_ALT, UCF_CKPT_DEF)
    ucf_model.eval()

    print("\n[2/3] ShanghaiTech  ST_Model")
    st_model = ST_Model(n_features=ST_FEAT_DIM, batch_size=32).to(DEVICE)
    load_checkpoint(st_model, ST_CKPT); st_model.eval()

    print("\n[3/3] XD-Violence  MILClassifier")
    xd_model = MILClassifier(input_dim=XD_FEAT_DIM, use_cbam=True, bank_size=256).to(DEVICE)
    load_checkpoint(xd_model, XD_CKPT); xd_model.eval()

    print("\nBuilding loaders ...")
    ucf_tr_norm = UCF_Normal_Loader(is_train=1, path=UCF_PATH, modality=UCF_MODALITY)
    ucf_tr_anom = UCF_Anomaly_Loader(is_train=1, path=UCF_PATH, modality=UCF_MODALITY)
    st_tr_norm  = ST_Dataset(args=st_args, is_normal=True,  test_mode=False)
    st_tr_anom  = ST_Dataset(args=st_args, is_normal=False, test_mode=False)
    try:
        from TTA_XD import Normal_Loader_XD, Anomaly_Loader_XD
        xd_tr_norm = Normal_Loader_XD(is_train=1, path=XD_DATA_PATH, augment=False)
        xd_tr_anom = Anomaly_Loader_XD(is_train=1, path=XD_DATA_PATH, augment=False)
    except Exception as e:
        print(f"  [WARN] XD train: {e}"); xd_tr_norm = xd_tr_anom = None

    ucf_anom_dl = DataLoader(UCF_Anomaly_Loader(is_train=0, path=UCF_PATH,
                             modality=UCF_MODALITY), batch_size=1, shuffle=False)
    ucf_norm_dl = DataLoader(UCF_Normal_Loader(is_train=0, path=UCF_PATH,
                             modality=UCF_MODALITY), batch_size=1, shuffle=False)
    st_test_dl  = DataLoader(ST_Dataset(args=st_args, is_normal=True, test_mode=True),
                             batch_size=1, shuffle=False, num_workers=0)
    xd_test_dl  = DataLoader(XDViolence_Loader(is_train=0, path=XD_DATA_PATH,
                             augment=False), batch_size=1, shuffle=False, num_workers=4)

    results = {}

    results['UCF->ST'] = run_pair(
        'UCF -> ST', model=ucf_model,
        src_dim=UCF_FEAT_DIM, tgt_dim=ST_FEAT_DIM, shape_fn=to_vad_shape,
        eval_fn=eval_st_style, eval_kwargs=dict(test_loader=st_test_dl, tta_steps=TTA_STEPS),
        tgt_normal_ds=st_tr_norm, tgt_anomaly_ds=st_tr_anom)

    results['ST->UCF'] = run_pair(
        'ST -> UCF', model=st_model,
        src_dim=ST_FEAT_DIM, tgt_dim=UCF_FEAT_DIM, shape_fn=to_st_shape,
        eval_fn=eval_ucf_style,
        eval_kwargs=dict(anomaly_loader=ucf_anom_dl, normal_loader=ucf_norm_dl, tta_steps=TTA_STEPS),
        tgt_normal_ds=ucf_tr_norm, tgt_anomaly_ds=ucf_tr_anom)

    results['UCF->XD'] = run_pair(
        'UCF -> XD', model=ucf_model,
        src_dim=UCF_FEAT_DIM, tgt_dim=XD_FEAT_DIM, shape_fn=to_vad_shape,
        eval_fn=eval_xd_style, eval_kwargs=dict(test_loader=xd_test_dl, tta_steps=TTA_STEPS),
        tgt_normal_ds=xd_tr_norm or ucf_tr_norm, tgt_anomaly_ds=xd_tr_anom or ucf_tr_anom)

    results['XD->UCF'] = run_pair(
        'XD -> UCF', model=xd_model,
        src_dim=XD_FEAT_DIM, tgt_dim=UCF_FEAT_DIM, shape_fn=to_xd_shape,
        eval_fn=eval_ucf_style,
        eval_kwargs=dict(anomaly_loader=ucf_anom_dl, normal_loader=ucf_norm_dl, tta_steps=TTA_STEPS),
        tgt_normal_ds=ucf_tr_norm, tgt_anomaly_ds=ucf_tr_anom)

    results['ST->XD'] = run_pair(
        'ST -> XD', model=st_model,
        src_dim=ST_FEAT_DIM, tgt_dim=XD_FEAT_DIM, shape_fn=to_st_shape,
        eval_fn=eval_xd_style, eval_kwargs=dict(test_loader=xd_test_dl, tta_steps=TTA_STEPS),
        tgt_normal_ds=xd_tr_norm or st_tr_norm, tgt_anomaly_ds=xd_tr_anom or st_tr_anom)

    results['XD->ST'] = run_pair(
        'XD -> ST', model=xd_model,
        src_dim=XD_FEAT_DIM, tgt_dim=ST_FEAT_DIM, shape_fn=to_xd_shape,
        eval_fn=eval_st_style, eval_kwargs=dict(test_loader=st_test_dl, tta_steps=TTA_STEPS),
        tgt_normal_ds=st_tr_norm, tgt_anomaly_ds=st_tr_anom)

    print("\n" + "="*58)
    print("  CROSS-DATASET TRANSFER — FINAL RESULTS")
    print("="*58)
    print(f"  {'Pair':<12}  {'AUC':>8}  {'AP':>8}")
    print("  " + "-"*34)
    for k, v in results.items():
        auc_s = f"{v['auc']*100:.2f}%" if v else "---"
        ap_s  = f"{v['ap']*100:.2f}%" if (v and 'ap' in v) else "   ---"
        print(f"  {k:<12}  {auc_s:>8}  {ap_s:>8}")
    print("="*58)

    def fmt(k):
        v = results.get(k); return f"{v['auc']*100:.1f}" if v else "---"
    print("\n  LaTeX row:")
    print(r"  \textbf{AnoTTA (Ours)} & "
          f"{fmt('UCF->ST')} & {fmt('ST->UCF')} & "
          f"{fmt('UCF->XD')} & {fmt('XD->UCF')} & "
          f"{fmt('ST->XD')} & {fmt('XD->ST')} \\\\")

    out = os.path.join(SAVE_DIR, 'cross_dataset_results.json')
    with open(out, 'w') as f: json.dump(results, f, indent=2)
    print(f"\n  Saved -> {out}\n")

if __name__ == '__main__':
    main()