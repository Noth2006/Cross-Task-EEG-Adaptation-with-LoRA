import torch
import torch.nn as nn
device = torch.device("cuda" if True else "cpu")

import os
import glob
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset,Subset, DataLoader
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



@torch.no_grad()
def apply_max_norm(model):
    w = model.depthwise.weight
    flat = w.reshape(w.shape[0], -1)
    norms = torch.linalg.vector_norm(flat, ord=2, dim=1, keepdim=True)
    scale = torch.clamp(norms, max=1.0) / (norms + 1e-8)
    w.mul_(scale.reshape(-1, 1, 1, 1))

    w = model.proj.weight
    norms = torch.linalg.vector_norm(w, ord=2, dim=1, keepdim=True)
    scale = torch.clamp(norms, max=0.25) / (norms + 1e-8)
    w.mul_(scale)




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



# paths are removed 
ROOT = Path("")
OUT = Path("")  
OUT.mkdir(parents=True, exist_ok=True)

subjects = [f"S{i:02d}" for i in range(1, 22)]
seeds = [51,52,53,54,55]
FINGERS = [1, 2, 4]   
max_epochs = 300
BATCH, LR, FACTOR, DROPOUT = 16, 1e-3, 0.5, 0.5
criterion = nn.CrossEntropyLoss()


def load_windowed(folder, trial_sec):
    ds = OfflineMovementDataset(path=str(folder), fs=1024, trial_sec=trial_sec, include_fingers=FINGERS)
    assert set(int(k) for k in ds.label_map) == set(FINGERS), f"unexpected labels in {folder}: {ds.label_map}"
    print(f"  {folder.name}: X {ds.X.shape}, class counts {np.bincount(ds.y)}")
    windowed = WindowedTrialDataset(ds.X, ds.y, fs=1024, target_fs=100, window_sec=1, step_samples=128)
    del ds
    return windowed


def train_one_seed(seed, off_windowed, s1_loader, ckpt_path):
    tid = off_windowed.trial_ids
    all_trials = np.unique(tid)
    rng = np.random.default_rng(seed)
    rng.shuffle(all_trials)
    val_trials = all_trials[: int(0.2 * len(all_trials))]
    val_mask = np.isin(tid, val_trials)
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]

    train_loader = DataLoader(Subset(off_windowed, train_idx), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(Subset(off_windowed, val_idx), batch_size=BATCH, shuffle=False)

    torch.manual_seed(seed)
    model = EEGNet(dropout=DROPOUT).to(device)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=FACTOR, patience=30)

    prev_epoch_loss, patience = float("inf"), 0
    best_val_acc, best_state, best_epoch = -1.0, None, -1

    for epoch in tqdm(range(max_epochs), desc=f"seed {seed}", leave=False,mininterval=20):
        model.train()
        for x, y, trial_id in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            apply_max_norm(model)

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
            patience = 0
            prev_epoch_loss = val_loss_sum

        if patience >= 80:
            break

    model.load_state_dict(best_state)
    torch.save(best_state, ckpt_path)

    val_mv = evaluate_with_majority_vote(model, val_loader, device)
    s1_acc = evaluate_with_majority_vote(model, s1_loader, device)
    return dict(seed=seed, best_val_win_acc=best_val_acc, best_epoch=best_epoch,
                val_maj_vote=val_mv, s1_base=s1_acc, stopped_epoch=epoch)



for subj in subjects:
    res_path = OUT / f"{subj}_results.json"
    if res_path.exists():                    
        print(f"{subj}: already done, skipping")
        continue

    off_dir = ROOT / subj / "OfflineMovement"
    s1_dir = ROOT / subj / "OnlineMovement_Sess01_3class_Base"
    if not (off_dir.is_dir() and s1_dir.is_dir()):
        print(f"{subj}: missing folder(s), skipping")
        continue

    
    off_windowed = load_windowed(off_dir, trial_sec=5)
    s1_windowed = load_windowed(s1_dir, trial_sec=3)

    s1_loader = DataLoader(s1_windowed, batch_size=32, shuffle=False)

    subj_results = []
    for seed in seeds:
        r = train_one_seed(seed, off_windowed, s1_loader, OUT / f"{subj}_me_offline_seed{seed}.pt")
        subj_results.append(r)
        print(f"{subj} seed {seed}: best val(win) {r['best_val_win_acc']*100:.2f}% @ {r['best_epoch']} | "
              f"val maj-vote {r['val_maj_vote']*100:.2f}% | S1 base {r['s1_base']*100:.2f}%")

    s1_accs = np.array([r["s1_base"] for r in subj_results])
    summary = dict(
        subject=subj,
        s1_base_mean=float(s1_accs.mean()),   
        s1_base_std=float(s1_accs.std()),
        seeds=subj_results,
    )
    with open(res_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{subj}: S1 base {s1_accs.mean()*100:.2f} +- {s1_accs.std()*100:.2f}% over {len(seeds)} seeds")

    del off_windowed, s1_windowed, s1_loader
    gc.collect()
    torch.cuda.empty_cache()


#summary

all_accs, subj_means = [], []
print("\n S1 base accuracy per subject (mean +- std over seeds)")
for subj in subjects:
    p = OUT / f"{subj}_results.json"
    if not p.exists():
        print(f"{subj}: no results")
        continue
    d = json.load(open(p))
    sd = d["seeds"] if isinstance(d, dict) else d      
    accs = np.array([r["s1_base"] for r in sd])
    if [r["seed"] for r in sd] != seeds:
        print(f"{subj}: note - seeds in this JSON {[r['seed'] for r in sd]} differ from current seeds")
    print(f"{subj}: {accs.mean()*100:.2f} +- {accs.std()*100:.2f}%")
    all_accs.extend(accs.tolist())
    subj_means.append(accs.mean())

if subj_means:
    print("\n Overall ")
    print(f"{len(subj_means)} subjects, {len(all_accs)} models total")
    print(f"Average S1 base accuracy over all subjects and seeds: {np.mean(all_accs)*100:.2f}%")
    print(f"Std across subjects (of per-subject means): {np.std(subj_means)*100:.2f}%")