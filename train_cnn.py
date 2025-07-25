import os
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import torchaudio.transforms as T

# ── 0. 공통 클래스 → 라벨 매핑 한 번만 만든다 ──────────────────
from pathlib import Path
from collections import Counter

base_dir = Path("mel_specs aiu")
SUFFIXES = ["_a", "_i", "_u"]
N_MELS = 128

# ‘normal_a’ → ‘normal’, ‘polyp_a’ → ‘polyp’ …
class_names = {
    d.name[: -len(sfx)]
    for d in base_dir.iterdir() if d.is_dir()
    for sfx in SUFFIXES if d.name.endswith(sfx)
}
class_to_idx = {name: idx for idx, name in enumerate(sorted(class_names))}
print("클래스 → 라벨:", class_to_idx)   # 예: {'normal': 0, 'polyp': 1, ...}

def collect_paths_by_suffix(base: Path, suffix: str):
    paths, labels, ids = [], [], []                     # ★ ID 목록도 함께
    for d in base.iterdir():
        if d.is_dir() and d.name.endswith(suffix):
            class_name = d.name[: -len(suffix)]
            lbl = class_to_idx[class_name]
            for f in d.rglob("*.npy"):
                paths.append(str(f))
                labels.append(lbl)
                ids.append(get_patient_id(f))           # ★ ID 추출
    return paths, labels, ids

# 1. 모델 클래스 불러오기
# (사전 준비: 'CNN mel spect.py' -> 'model_cnn.py'로 파일명 변경 권장)
try:
    from model_cnn import Mel2DCNN
except ImportError:
    print("[오류] 'model_cnn.py' 파일에서 'Mel2DCNN' 클래스를 찾을 수 없습니다.")
    print("모델 정의가 포함된 파일의 이름을 'model_cnn.py'로 변경했는지 확인해주세요.")
    exit()

# ---------------------------------
# 2. PyTorch Dataset 정의
# ---------------------------------
class SoundDataset(Dataset):
    """
    Mel-spectrogram .npy 파일들을 불러오기 위한 PyTorch Dataset 입니다.
    """
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        # .npy 파일에서 Mel-spectrogram 불러오기
        # shape: (n_mels, n_frames)
        mel_spec = np.load(path)
        
        # PyTorch 텐서로 변환
        mel_spec_tensor = torch.from_numpy(mel_spec).float()
        
        return mel_spec_tensor, label

def save_augmented_minority_pairwise(file_paths, labels, ids, suffixes,
                                     train_pids, minor_pairs, pair_cnt, max_cnt,
                                     n_aug_per_file=2,
                                     freq_param=13, time_param=15):
    """
    (label, suffix) 단위로 max_cnt까지 증강.
    pair_cnt: {(lbl, suf): 원본 개수}
    minor_pairs: 부족한 pair 집합
    """
    fmask = T.FrequencyMasking(freq_param)
    tmask = T.TimeMasking(time_param)

    per_pair_generated = Counter()
    aug_cnt = 0

    for path, lbl, pid, suf in zip(file_paths, labels, ids, suffixes):
        pair = (lbl, suf)

        # 1) Train 환자/소수 pair만 대상
        if pid not in train_pids or pair not in minor_pairs:
            continue

        # 2) 목표 수량 도달 시 skip
        if pair_cnt[pair] + per_pair_generated[pair] >= max_cnt:
            continue

        mel = np.load(path).astype(np.float32)
        stem, ext = os.path.splitext(path)

        for _ in range(n_aug_per_file):
            if pair_cnt[pair] + per_pair_generated[pair] >= max_cnt:
                break

            aug = tmask(fmask(torch.from_numpy(mel))).numpy()
            new_path = f"{stem}_aug{per_pair_generated[pair]}{ext}"

            if os.path.exists(new_path):
                per_pair_generated[pair] += 1
                continue

            np.save(new_path, aug)
            per_pair_generated[pair] += 1
            aug_cnt += 1

    print("쌍별 추가 수:", per_pair_generated)
    return aug_cnt


def clean_aug_files(root="mel_specs aiu", postfix="_aug"):
    removed = 0
    for f in Path(root).rglob(f"*{postfix}*.npy"):
        f.unlink()
        removed += 1
    print(f"🗑  기존 증강 파일 {removed}개 삭제 완료")
# ---------------------------------
# 3. 패딩을 위한 Custom Collate 함수
# ---------------------------------
def pad_collate_fn(batch):
    """
    길이가 다른 시퀀스들을 패딩하여 하나의 배치로 만듭니다.
    가장 긴 시퀀스 길이에 맞춰 나머지 시퀀스들의 뒷부분을 0으로 채웁니다.
    
    Args:
        batch: (spectrogram_tensor, label) 튜플의 리스트
    
    Returns:
        (padded_spectrograms_tensor, labels_tensor) 튜플
    """
    # 데이터를 시퀀스 길이에 따라 내림차순으로 정렬
    batch.sort(key=lambda x: x[0].shape[1], reverse=True)
    
    spectrograms, labels = zip(*batch)
    
    # 배치 내 가장 긴 시퀀스의 길이
    max_len = spectrograms[0].shape[1]
    
    # 모든 스펙트로그램을 max_len에 맞춰 패딩
    padded_spectrograms = []
    for spec in spectrograms:
        # spec의 shape: (n_mels, n_frames)
        # 두 번째 차원(n_frames)을 패딩해야 함
        pad_len = max_len - spec.shape[1]
        # F.pad는 마지막 차원부터 패딩을 적용. (왼쪽 패딩, 오른쪽 패딩)
        padded_spec = F.pad(spec, (0, pad_len), 'constant', 0)
        padded_spectrograms.append(padded_spec)
        
    # 패딩된 스펙트로그램들을 하나의 텐서로 합침
    spectrograms_tensor = torch.stack(padded_spectrograms)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return spectrograms_tensor, labels_tensor

# ---------------------------------
# 5. 훈련, 평가, 피처 추출 함수
# ---------------------------------
def train_one_epoch(model, data_loader, criterion, optimizer, device):
    """한 에폭(epoch) 동안 모델을 훈련합니다."""
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in tqdm(data_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        # 순전파
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 통계 기록
        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = total_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

from sklearn.metrics import classification_report # 파일 상단에 이 줄을 추가하세요.

def evaluate(model, data_loader, criterion, device):
    """모델을 평가하고, 예측값과 실제값을 반환합니다.""" # (설명 변경)
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    all_labels = [] # 실제 라벨을 저장할 리스트 (추가)
    all_preds = []  # 예측 라벨을 저장할 리스트 (추가)

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # --- ▼▼▼ 추가된 부분 ▼▼▼ ---
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            # --- ▲▲▲ 추가된 부분 ▲▲▲ ---

    epoch_loss = total_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    
    # 예측값과 실제값을 numpy 배열로 변환하여 반환 (변경)
    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds)

def get_feature_extractor(model_path, num_classes, n_mels, device):
    """
    학습된 모델을 불러와 마지막 분류 레이어를 제거한 피처 추출기 모델을 반환합니다.
    """
    # 먼저 전체 모델 구조를 생성합니다.
    model = Mel2DCNN(num_classes=num_classes, n_mels=n_mels)
    
    # 학습된 가중치를 불러옵니다.
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 마지막 분류 레이어(fc2)를 Identity 레이어로 교체합니다.
    # 이렇게 하면 fc1의 출력이 모델의 최종 출력이 됩니다.
    model.fc2 = torch.nn.Identity()
    
    model.to(device)
    model.eval() # 피처 추출은 항상 평가 모드에서 수행합니다.
    return model

def extract_features(feature_extractor, data_loader, device):
    """
    피처 추출기 모델을 사용하여 데이터셋 전체의 피처를 추출합니다.
    """
    all_features = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Extracting Features"):
            inputs = inputs.to(device)
            
            # 모델을 통과시켜 피처를 얻습니다.
            features = feature_extractor(inputs)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            
    # 리스트들을 하나의 numpy 배열로 합칩니다.
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_features, all_labels

import re
from pathlib import Path

ID_RE = re.compile(r"^(\d+)")     # 파일명 첫 연속 숫자

def get_patient_id(file_path: str | Path) -> str:
    """Path → Patient ID (str). 예: '1251372a-...' → '1251372'"""
    stem = Path(file_path).stem       # 확장자 제거
    m = ID_RE.match(stem)
    if not m:
        raise ValueError(f"ID 추출 실패: {file_path}")
    return m.group(1)




# ---------------------------------
# 메인 실행 블록
# ---------------------------------
if __name__ == '__main__':
    clean_aug_files()

    BASE_DATA_DIR = "mel_specs aiu"
    SUFFIXES      = ["_a", "_i", "_u"]
    MODEL_TMPL    = "best_cnn_model{}.pth"
    BATCH_SIZE = 16; NUM_EPOCHS = 20; LR = 1e-3; N_MELS = 128
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 1. 환자 분할 ----------
    all_paths_by_suf, all_labels_by_suf, all_ids_by_suf = {}, {}, {}
    all_paths, all_labels, all_ids, all_sufs = [], [], [], []

    for suf in SUFFIXES:
        p, l, i = collect_paths_by_suffix(Path(BASE_DATA_DIR), suf)
        all_paths_by_suf[suf], all_labels_by_suf[suf], all_ids_by_suf[suf] = p, l, i

        all_paths  += p
        all_labels += l
        all_ids    += i
        all_sufs   += [suf] * len(p)

    # 환자 대표 라벨 산출
    df = pd.DataFrame({"pid": all_ids, "label": all_labels})
    patient_major = df.groupby("pid")["label"].agg(lambda x: x.mode()[0])
    pids = patient_major.index.to_numpy()
    pid_labels = patient_major.to_numpy()

    # Stratified patient split (60/20/20)
    sss1 = StratifiedShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    trv_idx, te_idx = next(sss1.split(pids, pid_labels))
    pids_trv, pids_te = pids[trv_idx], pids[te_idx]

    labels_trv = patient_major.loc[pids_trv].to_numpy()
    sss2 = StratifiedShuffleSplit(test_size=0.25, n_splits=1, random_state=42)
    tr_idx, va_idx = next(sss2.split(pids_trv, labels_trv))
    pids_tr, pids_va = pids_trv[tr_idx], pids_trv[va_idx]

    train_pids, val_pids, test_pids = set(pids_tr), set(pids_va), set(pids_te)
    assert train_pids.isdisjoint(val_pids) and train_pids.isdisjoint(test_pids) and val_pids.isdisjoint(test_pids)
    print(f"✅ 환자 분할 완료: Train {len(train_pids)}, Val {len(val_pids)}, Test {len(test_pids)} 명")

    # ---------- 2. (label, suffix) 단위로 증강 기준 계산 ----------
    train_mask     = [get_patient_id(p) in train_pids for p in all_paths]
    train_labels   = [l for l, m in zip(all_labels, train_mask) if m]
    train_suffixes = [s for s, m in zip(all_sufs,  train_mask) if m]

    pair_cnt   = Counter(zip(train_labels, train_suffixes))
    max_cnt    = max(pair_cnt.values())
    minor_pairs = {k for k, v in pair_cnt.items() if v < max_cnt}

    print("pair_cnt:", pair_cnt)
    print("max_cnt :", max_cnt)
    print("minor_pairs:", minor_pairs)

    # ---------- 3. 증강 실행 ----------
    aug_num = save_augmented_minority_pairwise(
        all_paths, all_labels, all_ids, all_sufs,
        train_pids, minor_pairs, pair_cnt, max_cnt,
        n_aug_per_file=2, freq_param=13, time_param=15
    )
    print(f"🆕 소수 (label,suffix) 증강 파일 {aug_num}개 저장 완료")

    # 증강 후 경로/라벨/ID 재수집
    for suf in SUFFIXES:
        p, l, i = collect_paths_by_suffix(Path(BASE_DATA_DIR), suf)
        all_paths_by_suf[suf]  = p
        all_labels_by_suf[suf] = l
        all_ids_by_suf[suf]    = i

    # ---------- 4. 모음별 학습 루프 ----------
    for suf in SUFFIXES:
        print(f"\n================  [{suf}] 데이터 준비  ================")
        file_paths, labels, ids = all_paths_by_suf[suf], all_labels_by_suf[suf], all_ids_by_suf[suf]
        if not file_paths:
            print(f"[경고] '{suf}' 데이터가 없습니다."); continue

        tr_p, tr_l = [p for i, p in enumerate(file_paths) if ids[i] in train_pids], [l for i, l in enumerate(labels) if ids[i] in train_pids]
        va_p, va_l = [p for i, p in enumerate(file_paths) if ids[i] in val_pids],   [l for i, l in enumerate(labels) if ids[i] in val_pids]
        te_p, te_l = [p for i, p in enumerate(file_paths) if ids[i] in test_pids],  [l for i, l in enumerate(labels) if ids[i] in test_pids]
        print(f"샘플 분할: Train {len(tr_p)}, Val {len(va_p)}, Test {len(te_p)} 개")

        tr_ds = SoundDataset(tr_p, tr_l)
        va_ds = SoundDataset(va_p, va_l)
        te_ds = SoundDataset(te_p, te_l)

        # (선택) 완전 재현성: generator 지정
        g = torch.Generator().manual_seed(42)
        tr_loader = DataLoader(tr_ds, BATCH_SIZE, True,  collate_fn=pad_collate_fn, generator=g)
        va_loader = DataLoader(va_ds, BATCH_SIZE, False, collate_fn=pad_collate_fn)
        te_loader = DataLoader(te_ds, BATCH_SIZE, False, collate_fn=pad_collate_fn)

        model = Mel2DCNN(num_classes=len(class_to_idx), n_mels=N_MELS).to(DEVICE)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        best_acc = 0.0
        for epoch in range(NUM_EPOCHS):
            tr_loss, tr_acc = train_one_epoch(model, tr_loader, criterion, optimizer, DEVICE)
            va_loss, va_acc, _, _ = evaluate(model, va_loader, criterion, DEVICE)
            if va_acc > best_acc:
                best_acc = va_acc
                torch.save(model.state_dict(), MODEL_TMPL.format(suf))
            print(f"[{suf}][E{epoch+1}] train {tr_acc:.3f} | val {va_acc:.3f}")

        # --- 테스트 ---
        print(f"\n--- [{suf}] 최종 모델 상세 평가 ---")
        model.load_state_dict(torch.load(MODEL_TMPL.format(suf), map_location=DEVICE))
        te_loss, te_acc, y_true, y_pred = evaluate(model, te_loader, criterion, DEVICE)
        print(f"✅ [{suf}] 최종 test acc = {te_acc:.4f}")

        target_names = sorted(class_to_idx, key=class_to_idx.get)
        report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
        print("\n[Classification Report]")
        print(report)

        # --- 피처 추출 ---
        feature_extractor = get_feature_extractor(MODEL_TMPL.format(suf), len(class_to_idx), N_MELS, DEVICE)
        full_ds = SoundDataset(file_paths, labels)
        full_ld = DataLoader(full_ds, BATCH_SIZE, False, collate_fn=pad_collate_fn)
        feats, labs = extract_features(feature_extractor, full_ld, DEVICE)
        ids_all = [get_patient_id(p) for p in file_paths]

        np.save(f"cnn_features{suf}.npy", feats)
        np.save(f"cnn_labels{suf}.npy",   labs)
        np.save(f"cnn_ids{suf}.npy",      np.array(ids_all))
        print(f"[{suf}] 피처 shape {feats.shape} 저장 완료")

    np.savez(
        "patient_split_seed(agu).npz",
        train=list(train_pids),
        val=list(val_pids),
        test=list(test_pids)
    )
    print("✅ 환자 분할 결과 저장 완료 → patient_split_seed(agu).npz")
