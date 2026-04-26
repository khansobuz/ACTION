import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import json
import torch.nn.functional as F

# Import your dataset loaders
from dataset_xd import Normal_Loader_XD, Anomaly_Loader_XD, XDViolence_Loader

# ==================== SETTINGS - CHANGE THESE ====================
DATA_PATH = 'C:/Users/khanm/Desktop/lab_project/DMN/xd_vio/'
FEATURE_DIM = 1024   
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.0001
SAVE_DIR = './checkpoints'
USE_CBAM = True
USE_REPLAY_BUFFER = True
REPLAY_BUFFER_SIZE = 500
# AnoTTA settings
TTA_LR = 1e-5
ANOMALY_THRESHOLD = 0.5
TEMPORAL_WEIGHT = 0.1
MEMORY_WEIGHT = 0.1
# =================================================================


# ============================================================
# CONTRIBUTION 3: Scene-Adaptive Normal Memory Bank
# ============================================================
class SceneAdaptiveMemoryBank(nn.Module):
    def __init__(self, feature_dim=512, bank_size=256, momentum=0.995):
        super(SceneAdaptiveMemoryBank, self).__init__()
        self.bank_size = bank_size
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.register_buffer('memory', F.normalize(torch.randn(bank_size, feature_dim), dim=1))
        self.register_buffer('ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('is_initialized', torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def update(self, normal_features):
        """Update only from confirmed normal (low anomaly-score) features."""
        if normal_features.size(0) == 0:
            return
        features = F.normalize(normal_features.detach(), dim=1)
        batch_size = features.size(0)
        ptr = int(self.ptr)
        slots = min(batch_size, self.bank_size)
        indices = torch.arange(ptr, ptr + slots) % self.bank_size
        if not self.is_initialized:
            self.memory[indices] = features[:slots]
            self.is_initialized[0] = True
        else:
            self.memory[indices] = (
                self.momentum * self.memory[indices] +
                (1 - self.momentum) * features[:slots]
            )
        self.memory[indices] = F.normalize(self.memory[indices], dim=1)
        self.ptr[0] = (ptr + slots) % self.bank_size

    def compute_normality_score(self, features):
        features_norm = F.normalize(features, dim=1)
        memory_norm   = F.normalize(self.memory, dim=1)
        similarity = torch.matmul(features_norm, memory_norm.T)  # (B, M)
        topk = min(10, self.bank_size)
        topk_sim = similarity.topk(topk, dim=1)[0]
        return topk_sim.mean(dim=1)  # (B,)

    def bank_anchor_loss(self, normal_features):
        """Pull normal features toward stored normal prototypes."""
        if normal_features.size(0) == 0:
            return torch.tensor(0., device=normal_features.device)
        normality_score = self.compute_normality_score(normal_features)
        return (1.0 - normality_score).mean()


# ============================================================
# CONTRIBUTION 2: Temporal Consistency Regularization
# ============================================================
def temporal_consistency_loss(all_features):
    """
    all_features: (B, T, D)
    Penalizes large changes between consecutive frame representations.
    """
    if all_features.size(1) < 2:
        return torch.tensor(0., device=all_features.device)
    f_t  = all_features[:, :-1, :]
    f_t1 = all_features[:, 1:, :]
    consistency = F.mse_loss(f_t, f_t1.detach())
    if all_features.size(1) > 2:
        delta1 = all_features[:, 1:-1, :] - all_features[:, :-2, :]
        delta2 = all_features[:, 2:, :]   - all_features[:, 1:-1, :]
        smoothness = F.mse_loss(delta1, delta2.detach())
        return consistency + 0.5 * smoothness
    return consistency


 
class MambaBlock(nn.Module):
    """Temporal MambaBlock returning full sequence for temporal consistency."""
    def __init__(self, d_model, d_state=8, expand=1.5):
        super(MambaBlock, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.in_proj  = nn.Linear(d_model, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x_proj = self.in_proj(x)
        x_proj = self.activation(x_proj)
        x_conv = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.activation(x_conv)
        output = self.out_proj(x_conv)
        return output  # (B, T, d_model) — full sequence

 
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_t = x.transpose(1, 2)
        avg_out = self.fc(self.avg_pool(x_t).squeeze(-1))
        max_out = self.fc(self.max_pool(x_t).squeeze(-1))
        out = self.sigmoid(avg_out + max_out).unsqueeze(1)
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_t = x.transpose(1, 2)
        avg_out = torch.mean(x_t, dim=1, keepdim=True)
        max_out, _ = torch.max(x_t, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat)
        out = self.sigmoid(out)
        out = out.transpose(1, 2)
        return x * out


class CBAM(nn.Module):
    def __init__(self, channels, reduction=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


 
class ReplayBuffer:
    def __init__(self, capacity=500):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, features, label):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (features.cpu(), label)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None, None
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        features_list = []
        labels_list = []
        for idx in indices:
            feat, label = self.buffer[idx]
            features_list.append(feat)
            labels_list.append(label)
        return features_list, labels_list

    def __len__(self):
        return len(self.buffer)

 
class MILClassifier(nn.Module):
    def __init__(self, input_dim=1024, use_cbam=True, bank_size=256):
        super(MILClassifier, self).__init__()
        self.use_cbam = use_cbam

        # Feature embedding
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
        )

        # MambaBlock for temporal modeling (AnoTTA version)
        self.mamba_block = MambaBlock(d_model=512, d_state=8)

        # CBAM Attention
        if self.use_cbam:
            self.cbam = CBAM(channels=512, reduction=8, kernel_size=7)

        # Contribution 3: Scene-Adaptive Normal Memory Bank
        self.scene_memory = SceneAdaptiveMemoryBank(feature_dim=512, bank_size=bank_size)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        batch_size = x.size(0)
        seq_len    = x.size(1)

        # Extract features
        x_flat   = x.view(-1, x.size(-1))
        features = self.feature_extractor(x_flat)
        features = features.view(batch_size, seq_len, -1)  # (B, T, 512)

        # MambaBlock — returns full sequence (B, T, 512) for temporal consistency
        all_features = features + self.mamba_block(features)  # residual

        # CBAM
        if self.use_cbam:
            all_features = all_features + self.cbam(all_features)

        # Attention pooling
        attention_weights = self.attention(all_features)               # (B, T, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        weighted_features = (all_features * attention_weights).sum(dim=1)  # (B, 512)

        # Classification
        output = self.classifier(weighted_features)  # (B, 1)

        return output, attention_weights, all_features, weighted_features


# ============================================================
# CONTRIBUTION 1: Anomaly-Aware Selective TTA
# ============================================================
def selective_tta_update(model, inputs, tta_optimizer,
                          anomaly_threshold=0.5,
                          temporal_weight=0.1,
                          memory_weight=0.1):
    """
    AnoTTA test-time adaptation:
    C1 - Filter: adapt only on low anomaly-score (normal) frames
    C2 - Temporal consistency loss across consecutive frames
    C3 - Memory bank anchor loss + update with normal features only
    """
    model.train()
    with torch.enable_grad():
        output, attention_weights, all_features, weighted_features = model(inputs)

        # ---- C1: Confidence-guided selective filtering ----
        scores = output.squeeze(-1).detach()  # (B,)
        normal_mask = scores < anomaly_threshold

        if normal_mask.sum() == 0:
            # Fully anomalous batch — skip to prevent corruption
            model.eval()
            return

        normal_features = weighted_features[normal_mask]

        # Entropy minimization on selected normal samples only
        normal_out  = model.classifier(normal_features)
        normal_prob = normal_out.clamp(1e-7, 1 - 1e-7)
        entropy_loss = -(
            normal_prob * torch.log(normal_prob) +
            (1 - normal_prob) * torch.log(1 - normal_prob)
        ).mean()

        # ---- C2: Temporal Consistency Regularization ----
        temp_loss = temporal_consistency_loss(all_features)

        # ---- C3: Memory Bank Anchor Loss ----
        mem_loss = model.scene_memory.bank_anchor_loss(normal_features)

        total_loss = (
            entropy_loss +
            temporal_weight * temp_loss +
            memory_weight  * mem_loss
        )

        tta_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        tta_optimizer.step()

        # Update memory bank with confirmed normal features (C3)
        model.scene_memory.update(normal_features)

    model.eval()

 
def custom_collate_fn(batch):
    max_len  = max([item.shape[0] for item in batch])
    feat_dim = batch[0].shape[1]
    padded_batch = []
    for item in batch:
        seq_len = item.shape[0]
        if seq_len < max_len:
            padding = torch.zeros(max_len - seq_len, feat_dim)
            padded_item = torch.cat([item, padding], dim=0)
        else:
            padded_item = item
        padded_batch.append(padded_item)
    return torch.stack(padded_batch)
 
def train_epoch(model, normal_loader, anomaly_loader, optimizer, device,
                replay_buffer_normal=None, replay_buffer_anomaly=None, use_replay=False):
    model.train()
    total_loss = 0
    num_batches = min(len(normal_loader), len(anomaly_loader))

    normal_iter  = iter(normal_loader)
    anomaly_iter = iter(anomaly_loader)

    pbar = tqdm(range(num_batches), desc="Training")

    for _ in pbar:
        try:
            normal_features  = next(normal_iter)
            anomaly_features = next(anomaly_iter)
        except StopIteration:
            break

        normal_features  = normal_features.to(device)
        anomaly_features = anomaly_features.to(device)
        batch_size = normal_features.size(0)

        normal_scores,  _, normal_all_feat,  _ = model(normal_features)
        anomaly_scores, _, anomaly_all_feat, _ = model(anomaly_features)

        normal_labels  = torch.zeros(batch_size, 1).to(device)
        anomaly_labels = torch.ones(batch_size, 1).to(device)

        if use_replay and replay_buffer_normal is not None and replay_buffer_anomaly is not None:
            for i in range(batch_size):
                if normal_scores[i].item() > 0.5:
                    replay_buffer_normal.push(normal_features[i], 0)
            for i in range(batch_size):
                if anomaly_scores[i].item() < 0.5:
                    replay_buffer_anomaly.push(anomaly_features[i], 1)

        loss_normal  = nn.functional.binary_cross_entropy(normal_scores,  normal_labels)
        loss_anomaly = nn.functional.binary_cross_entropy(anomaly_scores, anomaly_labels)

        # Contribution 2: temporal consistency auxiliary loss during training
        temp_loss = (
            temporal_consistency_loss(normal_all_feat) +
            temporal_consistency_loss(anomaly_all_feat)
        ) * 0.5

        loss = loss_normal + loss_anomaly + 0.05 * temp_loss

        if use_replay and len(replay_buffer_normal) > 0 and len(replay_buffer_anomaly) > 0:
            replay_batch_size = min(8, batch_size // 2)
            replay_normal_feats,  replay_normal_labels  = replay_buffer_normal.sample(replay_batch_size)
            replay_anomaly_feats, replay_anomaly_labels = replay_buffer_anomaly.sample(replay_batch_size)

            if replay_normal_feats and replay_anomaly_feats:
                def pad_features(feat_list):
                    max_len = max([f.shape[0] for f in feat_list])
                    padded = []
                    for f in feat_list:
                        if f.shape[0] < max_len:
                            padding = torch.zeros(max_len - f.shape[0], f.shape[1])
                            padded.append(torch.cat([f, padding], dim=0))
                        else:
                            padded.append(f)
                    return torch.stack(padded)

                replay_normal_feats  = pad_features(replay_normal_feats).to(device)
                replay_anomaly_feats = pad_features(replay_anomaly_feats).to(device)

                actual_normal_batch  = replay_normal_feats.size(0)
                actual_anomaly_batch = replay_anomaly_feats.size(0)

                replay_normal_scores,  _, _, _ = model(replay_normal_feats)
                replay_anomaly_scores, _, _, _ = model(replay_anomaly_feats)

                replay_normal_labels_t  = torch.zeros(actual_normal_batch,  1).to(device)
                replay_anomaly_labels_t = torch.ones(actual_anomaly_batch, 1).to(device)

                replay_loss_normal  = nn.functional.binary_cross_entropy(replay_normal_scores,  replay_normal_labels_t)
                replay_loss_anomaly = nn.functional.binary_cross_entropy(replay_anomaly_scores, replay_anomaly_labels_t)

                loss = loss + 0.5 * (replay_loss_normal + replay_loss_anomaly)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches


# ============================================================
# evaluate — with AnoTTA selective adaptation at test time
# ============================================================
def evaluate(model, test_loader, device,
             use_tta=True, tta_lr=TTA_LR,
             anomaly_threshold=ANOMALY_THRESHOLD):
    model.eval()
    all_video_scores = []
    all_video_labels = []

    # TTA optimizer: norm layers only — lightweight
    if use_tta:
        tta_params = [p for n, p in model.named_parameters()
                      if 'norm' in n or 'bn' in n]
        if len(tta_params) == 0:
            tta_params = list(model.attention.parameters())
        tta_optimizer = optim.Adam(tta_params, lr=tta_lr, betas=(0.9, 0.999))

    with torch.no_grad():
        for features, labels, num_frames in tqdm(test_loader, desc="Evaluating"):
            features = features.to(device)

            # AnoTTA: all 3 contributions active at test time
            if use_tta:
                selective_tta_update(
                    model, features, tta_optimizer,
                    anomaly_threshold=anomaly_threshold,
                    temporal_weight=TEMPORAL_WEIGHT,
                    memory_weight=MEMORY_WEIGHT
                )

            video_score, attention_weights, _, _ = model(features)
            video_score = video_score.squeeze().cpu().item()
            video_label = 1.0 if labels.sum().item() > 0 else 0.0

            all_video_scores.append(video_score)
            all_video_labels.append(video_label)

    all_video_scores = np.array(all_video_scores)
    all_video_labels = np.array(all_video_labels)

    auc = roc_auc_score(all_video_labels, all_video_scores)
    ap  = average_precision_score(all_video_labels, all_video_scores)

    return auc, ap


 
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("\n=== Loading Training Data ===")
    normal_train  = Normal_Loader_XD(is_train=1, path=DATA_PATH, augment=False)
    anomaly_train = Anomaly_Loader_XD(is_train=1, path=DATA_PATH, augment=False)

    print("\n=== Loading Testing Data ===")
    test_dataset = XDViolence_Loader(is_train=0, path=DATA_PATH, augment=False)

    normal_train_loader = DataLoader(
        normal_train, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, collate_fn=custom_collate_fn, drop_last=True
    )
    anomaly_train_loader = DataLoader(
        anomaly_train, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, collate_fn=custom_collate_fn, drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    print("\n=== Creating Model ===")
    print(f"CBAM Attention: {'ENABLED' if USE_CBAM else 'DISABLED'}")
    print(f"Replay Buffer:  {'ENABLED' if USE_REPLAY_BUFFER else 'DISABLED'}")
    print(f"AnoTTA (C1+C2+C3): ENABLED")

    model = MILClassifier(
        input_dim=FEATURE_DIM,
        use_cbam=USE_CBAM,
        bank_size=256
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    replay_buffer_normal  = None
    replay_buffer_anomaly = None
    if USE_REPLAY_BUFFER:
        replay_buffer_normal  = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)
        replay_buffer_anomaly = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)
        print(f"Replay buffer size: {REPLAY_BUFFER_SIZE}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    print("\n=== Starting Training ===")
    best_ap  = 0
    best_auc = 0

    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        print(f"{'='*60}")

        train_loss = train_epoch(
            model, normal_train_loader, anomaly_train_loader, optimizer, device,
            replay_buffer_normal, replay_buffer_anomaly, USE_REPLAY_BUFFER
        )
        print(f"Train Loss: {train_loss:.4f}")

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            print("\nEvaluating...")
            auc, ap = evaluate(model, test_loader, device,
                               use_tta=True, tta_lr=TTA_LR,
                               anomaly_threshold=ANOMALY_THRESHOLD)
            print(f"AUC: {auc:.4f}, AP: {ap:.4f}")

            if ap > best_ap:
                best_ap  = ap
                best_auc = auc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'ap': ap,
                    'auc': auc,
                }, os.path.join(SAVE_DIR, 'best_model.pth'))
                print(f"✓ New best AP: {best_ap:.4f}")

        scheduler.step()
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best AUC: {best_auc:.4f}")
    print(f"Best AP:  {best_ap:.4f}")
    print("="*60)

    results = {
        'best_auc': float(best_auc),
        'best_ap':  float(best_ap),
        'use_cbam': USE_CBAM,
        'use_replay_buffer': USE_REPLAY_BUFFER,
        'anotta_contributions': ['selective_adaptation', 'temporal_consistency', 'scene_memory_bank']
    }
    with open(os.path.join(SAVE_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()