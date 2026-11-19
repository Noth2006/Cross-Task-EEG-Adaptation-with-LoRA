import torch
import torch.nn as nn
device = torch.device("cuda" if True else "cpu")

import os
import glob
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, Subset, DataLoader
from scipy.signal import resample, butter, lfilter
from scipy.stats import zscore
import gc, json, copy
from pathlib import Path
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm



class OfflineMovementDataset(Dataset):
    def __init__(self, path, fs=1024, trial_sec=5, include_fingers=None):
        self.fs = fs
        self.trial_sec = trial_sec
        self.max_samples = int(fs * trial_sec)
        self.include_fingers = include_fingers
        self.X, self.y = self._load(path)

    def _load_one_file(self, fname):
        d = sio.loadmat(fname)
        eeg = d["eeg"][0, 0]
        signal = eeg["data"].astype(np.float64)
        event = d["event"]

        start_idx, end_idx, labels = [], [], []

        for i in range(event.shape[1]):
            e = event[0, i]
            event_type = e["type"][0]
            if isinstance(event_type, np.ndarray): 
                event_type = event_type.item()
            event_type = str(event_type)

            sample = int(np.asarray(e["sample"]).squeeze()) # converting trial start to an int, using squeeze since its a nested np array
            value = int(np.asarray(e["value"]).squeeze()) # converting label value to int

            if event_type == "Target":
                start_idx.append(sample - 1)  
                labels.append(value)
            elif event_type == "TrialEnd":
                end_idx.append(sample - 1)

        X, y = [], []
        for start, end, label in zip(start_idx, end_idx, labels):
            if self.include_fingers is not None and label not in self.include_fingers:
                continue

            trial = signal[:, start:end]
            trial = trial[:, :self.max_samples] 

            if trial.shape[1] < self.max_samples:
                padding = self.max_samples - trial.shape[1]
                trial = np.pad(
                    trial, ((0, 0), (0, padding)),
                    mode="constant", constant_values=np.nan)

            # Common-average reference
            trial = trial - np.nanmean(trial, axis=0, keepdims=True)

            X.append(trial)
            y.append(label)

        return X, y

    def _load(self, path):
        if os.path.isfile(path):
            files = [path]
        elif os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*.mat")))

        all_X, all_y = [], []
        for fname in files:
            X, y = self._load_one_file(fname)
            all_X.extend(X)
            all_y.extend(y)

        X = np.stack(all_X).astype(np.float32)
        original_y = np.asarray(all_y, dtype=np.int64)

        unique_labels = sorted(np.unique(original_y))
        self.label_map = {original: new for new, original in enumerate(unique_labels)}

        y = np.asarray([self.label_map[v] for v in original_y], dtype=np.int64)
        return X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)




class WindowedTrialDataset(Dataset):
    def __init__(self, X, y, fs=1024, target_fs=100, window_sec=1, step_samples=128):
        self.windows = []
        self.labels = []
        self.trial_ids = []

        segment_size = int(window_sec * fs)          # 1024
        desired_len = int(window_sec * target_fs)     # 100

        b, a = butter(4, [4, 40], btype="bandpass", fs=target_fs)

        for trial_idx in range(len(X)):
            trial = X[trial_idx]

            for start in range(0, trial.shape[1] - segment_size + 1, step_samples):
                end = start + segment_size
                window = trial[:, start:end]

                if np.isnan(window).any():
                    continue

                window = resample(window, desired_len, axis=1)

                padding_length = 100
                padded = np.pad(
                    window, ((0, 0), (padding_length, padding_length)),
                    mode="constant", constant_values=0
                )

                filtered = lfilter(b, a, padded, axis=-1)
                filtered = filtered[:, padding_length:-padding_length]

                filtered = zscore(filtered, axis=1, nan_policy="omit")
                if np.isnan(filtered).any():
                    continue

                self.windows.append(filtered.astype(np.float32))
                self.labels.append(y[trial_idx])
                self.trial_ids.append(trial_idx)

        self.windows = np.stack(self.windows).astype(np.float32)
        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.trial_ids = np.asarray(self.trial_ids, dtype=np.int64)

        print("Window dataset:", self.windows.shape)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.windows[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long),
            self.trial_ids[idx]
        )



import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(
        self, n_channels=128, n_classes=3,n_samples=100,
        F1=8, D=2, F2=16, kernel_length=32, dropout=0.5):

        super().__init__()

        self.temp_conv = nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(F1, eps=1e-3, momentum=0.01)

        self.depthwise = nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D, eps=1e-3, momentum=0.01)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout)


        self.sep_depthwise = nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), groups=F1 * D,padding="same", bias=False)
        self.sep_pointwise = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2, eps=1e-3, momentum=0.01)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(dropout)

        self.flatten = nn.Flatten()

        self.proj = nn.Linear(F2 * 3, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.temp_conv(x)
        x = self.bn1(x)

        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.flatten(x)
        return self.proj(x)



class lora_point_wise_conv(nn.Module):

    def __init__(self,base_conv: nn.Conv2d, rank=4, alpha=4):

        super().__init__()
        self.base_conv = base_conv

        for p in self.base_conv.parameters():
            p.requires_grad = False

        self.alpha = alpha
        self.rank = rank

        c_out, c_in = base_conv.out_channels, base_conv.in_channels

        self.A = nn.Parameter(torch.empty(self.rank, c_in)) # r x 16
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

        self.B = nn. Parameter(torch.zeros(c_out,self.rank)) # 16 x r

    def forward(self,x):

        base_out = self.base_conv(x)
        B_, C, H, W = x.shape
        x_flat = x.reshape(B_, C, H * W) # B,16,25
        down = torch.matmul(self.A, x_flat) # B,r,25
        up = torch.matmul(self.B,down) # B,16,25
        up = up.reshape(B_, -1, H, W) # B,16,1,25

        return base_out + (self.alpha / self.rank) * up




class lora_linear(nn.Module):
    def __init__(self, base_linear: nn.Linear, alpha=2, rank=2):

        super().__init__()
        self.base_linear = base_linear
        for p in self.base_linear.parameters():
            p.requires_grad = False

        self.alpha = alpha
        self.rank = rank

        d_in, d_out = base_linear.in_features, base_linear.out_features

        self.A = nn.Parameter(torch.empty(rank, d_in))   # r x 48
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.B = nn.Parameter(torch.zeros(d_out, rank))  # 3 x r

    def forward(self, x):                                # B, 48
        base_out = self.base_linear(x)                   # B, 3
        down = x @ self.A.T                              # B, r
        up = down @ self.B.T                             # B, 3
        return base_out + (self.alpha / self.rank) * up




def build_lora_model(checkpoint_path, rank=4, alpha=4):
    model = EEGNet(
        n_channels=128, n_samples=100, n_classes=3,
        F1=8, D=2, F2=16, kernel_length=32, dropout=0.5
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    for p in model.parameters():
        p.requires_grad = False

    model.sep_pointwise = lora_point_wise_conv(model.sep_pointwise, rank=rank, alpha=alpha)
    model.proj = lora_linear(model.proj, rank=2, alpha=2)

    return model.to(device)



def evaluate_with_majority_vote(model, test_loader, device):
    model.eval()
    trial_votes = {}
    trial_true = {}

    with torch.no_grad():
        for x, y, trial_id in test_loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().numpy()

            for p, true_label, tid in zip(preds, y.numpy(), trial_id.numpy()):
                trial_votes.setdefault(tid, []).append(p)
                trial_true[tid] = true_label

    correct = 0
    for tid, votes in trial_votes.items():
        majority_pred = np.bincount(votes).argmax()
        if majority_pred == trial_true[tid]:
            correct += 1

    accuracy = correct / len(trial_votes)
    return accuracy




# paths removed
ROOT = Path("")
ME_DIR = Path("")       
OUT = Path("")      
OUT.mkdir(parents=True, exist_ok=True)

subjects = [f"S{i:02d}" for i in range(1, 22)]
seeds = [51, 52, 53, 54, 55]         
FINGERS = [1, 2, 4]                   
max_epochs = 300
LR, BATCH = 1e-3, 128
RANK, ALPHA = 4, 4
criterion = nn.CrossEntropyLoss()


def load_windowed(folder, trial_sec):
    ds = OfflineMovementDataset(path=str(folder), fs=1024, trial_sec=trial_sec, include_fingers=FINGERS)
    assert set(int(k) for k in ds.label_map) == set(FINGERS), f"unexpected labels in {folder}: {ds.label_map}"
    print(f"  {folder.name}: X {ds.X.shape}, class counts {np.bincount(ds.y)}")
    windowed = WindowedTrialDataset(ds.X, ds.y, fs=1024, target_fs=100, window_sec=1, step_samples=128)
    del ds
    return windowed


def lora_one_seed(subj, seed, off_windowed, s1_loader):
    tid = off_windowed.trial_ids
    all_trials = np.unique(tid)
    rng = np.random.default_rng(seed)
    rng.shuffle(all_trials)
    val_trials = all_trials[: int(0.2 * len(all_trials))]
    val_mask = np.isin(tid, val_trials)
    train_idx, val_idx = np.where(~val_mask)[0], np.where(val_mask)[0]
    train_loader = DataLoader(Subset(off_windowed, train_idx), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(Subset(off_windowed, val_idx), batch_size=BATCH, shuffle=False)

    torch.manual_seed(seed)
    model = build_lora_model(str(ME_DIR / f"{subj}_me_offline_seed{seed}.pt"), rank=RANK, alpha=ALPHA)

    names = sorted(n for n, p in model.named_parameters() if p.requires_grad)
    assert names == ["proj.A", "proj.B", "sep_pointwise.A", "sep_pointwise.B"], names

    me_only_acc = evaluate_with_majority_vote(model, s1_loader, device)

    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=30)
    prev_epoch_loss, patience = float("inf"), 0
    best_val_acc, best_state, best_epoch = -1.0, None, -1

    for epoch in tqdm(range(max_epochs), desc=f"{subj} LoRA seed {seed}", leave=False, mininterval=20):
        model.train()
        for m in (model.bn1, model.bn2, model.bn3):  
            m.eval()
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_sum, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss_sum += criterion(out, y).item() * len(y)
                correct += (out.argmax(1) == y).sum().item()
                n += len(y)
        val_acc = correct / n
        scheduler.step(val_loss_sum / n)

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
        if prev_epoch_loss <= val_loss_sum:
            patience += 1
        else:
            patience, prev_epoch_loss = 0, val_loss_sum
        if patience >= 80:
            break

    model.load_state_dict(best_state)
    torch.save(best_state, OUT / f"{subj}_mi_lora_seed{seed}.pt")
    lora_acc = evaluate_with_majority_vote(model, s1_loader, device)
    return dict(seed=seed, best_val_win_acc=best_val_acc, best_epoch=best_epoch, stopped_epoch=epoch,
                s1_me_only=me_only_acc, s1_lora=lora_acc)


for subj in subjects:
    res_path = OUT / f"{subj}_results.json"
    if res_path.exists():
        print(f"{subj}: already done, skipping")
        continue

    off_dir = ROOT / subj / "OfflineImagery"
    s1_dir = ROOT / subj / "OnlineImagery_Sess01_3class_Base"
    me_ckpts = [ME_DIR / f"{subj}_me_offline_seed{s}.pt" for s in seeds]
    if not (off_dir.is_dir() and s1_dir.is_dir()):
        print(f"{subj}: missing data folder(s), skipping")
        continue
    if not all(p.exists() for p in me_ckpts):
        print(f"{subj}: missing ME checkpoint(s), skipping")
        continue

    print(f"\n {subj} ")
    off_windowed = load_windowed(off_dir, trial_sec=5)
    s1_windowed = load_windowed(s1_dir, trial_sec=3)
    s1_loader = DataLoader(s1_windowed, batch_size=32, shuffle=False)

    subj_results = []
    for seed in seeds:
        r = lora_one_seed(subj, seed, off_windowed, s1_loader)
        subj_results.append(r)
        print(f"{subj} seed {seed}: val {r['best_val_win_acc']*100:.2f}% @ {r['best_epoch']} | "
              f"S1 MI: ME-only {r['s1_me_only']*100:.2f}% -> LoRA {r['s1_lora']*100:.2f}%")

    me = np.array([r["s1_me_only"] for r in subj_results])
    lo = np.array([r["s1_lora"] for r in subj_results])
    summary = dict(subject=subj,
                   s1_me_only_mean=float(me.mean()), s1_me_only_std=float(me.std()),
                   s1_lora_mean=float(lo.mean()), s1_lora_std=float(lo.std()),
                   seeds=subj_results)
    with open(res_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{subj}: ME-only {me.mean()*100:.2f} +- {me.std()*100:.2f}% | LoRA {lo.mean()*100:.2f} +- {lo.std()*100:.2f}%")

    del off_windowed, s1_windowed, s1_loader
    gc.collect()
    torch.cuda.empty_cache()



all_me, all_lo, sub_me, sub_lo = [], [], [], []
print("\n S1 base MI accuracy per subject (mean over seeds)")
for subj in subjects:
    p = OUT / f"{subj}_results.json"
    if not p.exists():
        print(f"{subj}: no results")
        continue
    d = json.load(open(p))
    me = np.array([r["s1_me_only"] for r in d["seeds"]])
    lo = np.array([r["s1_lora"] for r in d["seeds"]])
    if [r["seed"] for r in d["seeds"]] != seeds:
        print(f"{subj}: note - seeds in this JSON differ from current seeds")
    print(f"{subj}: ME-only {me.mean()*100:.2f} +- {me.std()*100:.2f}% | LoRA {lo.mean()*100:.2f} +- {lo.std()*100:.2f}%")
    all_me.extend(me.tolist()); all_lo.extend(lo.tolist())
    sub_me.append(me.mean()); sub_lo.append(lo.mean())

if sub_lo:
    print("\n Overall ")
    print(f"{len(sub_lo)} subjects, {len(all_lo)} models total")
    print(f"ME-only (no adaptation): {np.mean(all_me)*100:.2f}%  | std across subjects {np.std(sub_me)*100:.2f}%")
    print(f"ME + LoRA:               {np.mean(all_lo)*100:.2f}%  | std across subjects {np.std(sub_lo)*100:.2f}%")
    print(f"Mean per-subject gain (LoRA - ME-only): {(np.mean(sub_lo) - np.mean(sub_me))*100:+.2f} points")

