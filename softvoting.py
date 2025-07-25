import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


class ReencodeXGB(XGBClassifier):
    """fit 때 y 를 0‑based로 재인코딩 → 라벨 누락에 상관없이 동작"""
    def fit(self, X, y, **kw):
        self._le = LabelEncoder().fit(y)
        y_enc = self._le.transform(y)          # 0‑based
        return super().fit(X, y_enc, **kw)

    def predict(self, X):
        return self._le.inverse_transform(super().predict(X))

    def predict_proba(self, X):
        proba = super().predict_proba(X)
        # 컬럼 수가 self._le.classes_ 길이와 같지 않으면 패딩
        if proba.shape[1] != len(self._le.classes_):
            full = np.zeros((proba.shape[0], len(self._le.classes_)))
            full[:, self._le.transform(self._le.classes_)] = proba
            proba = full
        return proba



def get_specificity(y_true, y_pred, n_classes):
        """
        클래스별 Specificity = TN / (TN + FP) 계산
        반환 shape: (n_classes,)
        """
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
        tn = []
        fp = cm.sum(axis=0) - np.diag(cm)
        for k in range(n_classes):
            # 전체 합 – (해당 클래스 행합 + 해당 클래스 열합 – TP)
            tn_k = cm.sum() - (cm[k, :].sum() + cm[:, k].sum() - cm[k, k])
            tn.append(tn_k)
        tn = np.array(tn)
        return tn / (tn + fp + 1e-8)          # 0‑division 방지용 eps

def print_full_report(y_true, y_pred, target_names):
    """
    하나의 표에 Sensitivity, Specificity, TP, FP, FN, TN을 모두 출력합니다.
    """
    n_cls = len(target_names)
    
    # 1. classification_report에서 기본 정보 가져오기
    rep = classification_report(
        y_true, y_pred, target_names=target_names, labels=np.arange(n_cls),
        output_dict=True, zero_division=0)
    df = pd.DataFrame(rep).T.iloc[:n_cls, :]
    
    # 2. Confusion Matrix 기반 값들 계산
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_cls))
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)
    
    # 3. DataFrame에 새로운 열 추가
    df['Sensitivity'] = df['recall'] # recall -> Sensitivity 이름 변경
    df['Specificity'] = tn / (tn + fp + 1e-8)
    df['TP'] = tp
    df['FP'] = fp
    df['FN'] = fn
    df['TN'] = tn

    # 4. 최종 컬럼 순서 지정 및 출력
    df = df[['Sensitivity', 'Specificity', 'TP', 'FP', 'FN', 'TN', 'support']]
    print(df.round(3))
# ── 0. 파일 로드 및 설정 (수정) ────────────────────────────────
ORIGINAL_DATA_DIR = Path("mel_specs_augmented_aiu")
if not ORIGINAL_DATA_DIR.is_dir():
    ORIGINAL_DATA_DIR = Path("mel_specs aiu")

# 클래스 이름 -> 라벨 인덱스 맵 생성
SUFFIXES_FOR_MAP = ["_a", "_i", "_u"]
class_names = sorted({d.name[:-len(sfx)] for d in ORIGINAL_DATA_DIR.iterdir() if d.is_dir() for sfx in SUFFIXES_FOR_MAP if d.name.endswith(sfx)})
label_map = {name: idx for idx, name in enumerate(class_names)}
print("감지된 클래스:", label_map)


# ── 1. 환자 단위 분할 정보 로드 ──────────────────────────────────
# CNN 학습 시 사용했던 환자 분할 정보를 그대로 사용
split_file = "patient_split_seed(agu).npz"
try:
    sp = np.load(split_file, allow_pickle=True)
    train_pids = set(sp["train"])
    val_pids   = set(sp["val"])
    test_pids  = set(sp["test"])
    print(f"✅ CNN과 동일한 환자 분할 정보 로드 완료: Train {len(train_pids)}, Val {len(val_pids)}, Test {len(test_pids)} 명")
except FileNotFoundError:
    print(f"[오류] 환자 분할 파일 '{split_file}'을 찾을 수 없습니다. train_cnn.py를 먼저 실행하여 파일을 생성해야 합니다.")
    exit()

# 교집합 검증
assert train_pids.isdisjoint(val_pids) and train_pids.isdisjoint(test_pids) and val_pids.isdisjoint(test_pids)


# ── 2. 모음별 스케일러 + XGB(GridSearch) 학습 (데이터 로딩 로직 통합) ──────────────
PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth":    [5, 7],
    "learning_rate":[0.05, 0.1]
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

models, scalers = {}, {}
all_y_true = None  # 모든 모음에서 동일한 test set의 y_true를 공유하기 위한 변수
all_probas = []    # 소프트 보팅을 위한 확률 저장 리스트

for sfx in ("a", "i", "u"):
    print(f"\n================ Processing Vowel: '{sfx}' ================")
    
    # 1. 각 모음에 해당하는 피처, ID, 라벨 파일 불러오기
    try:
        X = np.load(f"cnn_features_{sfx}.npy")
        ids = np.load(f"cnn_ids_{sfx}.npy")
        y = np.load(f"cnn_labels_{sfx}.npy")
    except FileNotFoundError:
        print(f"[경고] '{sfx}'에 대한 피처/ID/라벨 파일이 없습니다. 건너<binary data, 2 bytes>니다.")
        continue

    # 2. 환자 분할 기준(pids)을 이용해 현재 모음 데이터의 인덱스 생성
    # 이 방식은 'a'에만 있는 샘플, 'i'에만 있는 샘플 등을 모두 올바르게 처리합니다.
    idx_tr = np.where(np.isin(ids, list(train_pids)))[0]
    idx_va = np.where(np.isin(ids, list(val_pids)))[0]
    idx_te = np.where(np.isin(ids, list(test_pids)))[0]

    print(f"샘플 분할: Train {len(idx_tr)}, Val {len(idx_va)}, Test {len(idx_te)} 개")
    
    # 클래스 분포 확인 (디버깅용)
    uniq, cnt = np.unique(y[idx_te], return_counts=True)
    print(f"Test set class distribution for '{sfx}': {dict(zip(uniq, cnt))}")

    # 3. 스케일링 및 모델 학습 (기존 코드와 유사)
    scaler = StandardScaler().fit(X[idx_tr])
    X_sc   = scaler.transform(X)

    xgb = ReencodeXGB(
        eval_metric="mlogloss",
        tree_method="hist",
        device=DEVICE,
        random_state=42
    )

    cv_splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    gscv = GridSearchCV(
        estimator=xgb,
        param_grid=PARAM_GRID,
        cv=cv_splitter,
        scoring="accuracy",
        n_jobs=-1,
        error_score="raise"
    )
    gscv.fit(X_sc[idx_tr], y[idx_tr], groups=ids[idx_tr])

    best = gscv.best_estimator_
    val_acc = accuracy_score(y[idx_va], best.predict(X_sc[idx_va]))
    print(f"'{sfx}' best {gscv.best_params_} | val acc {val_acc:.3f}")

    models[sfx], scalers[sfx] = best, scaler
    
    # --- 단일 모델 테스트 성능 ---
    print(f"\n--- 단일 모델 '{sfx}' 테스트 성능 ---")
    y_pred_single = best.predict(X_sc[idx_te])
    y_true_single = y[idx_te]
    acc_single = accuracy_score(y_true_single, y_pred_single)
    print(f"Accuracy: {acc_single:.4f}")
    print_full_report(y_true_single, y_pred_single, list(label_map.keys()))

    # 소프트 보팅을 위해 test set 예측 확률 저장
    # y_true는 모든 모음에서 동일해야 함 (환자 분할이 같으므로)
    if all_y_true is None:
        all_y_true = y[idx_te]
    
    # all_probas 리스트에 각 모델의 예측 확률을 추가
    all_probas.append(best.predict_proba(X_sc[idx_te]))


# ── 3. 소프트 보팅(확률 평균) ───────────────────────────────
# all_probas에 저장된 모든 확률을 합산
proba_sum = np.sum(all_probas, axis=0) 
y_pred = proba_sum.argmax(axis=1)
print("\n\n📊 --- 최종 Soft-Voting Result ---")
print("Accuracy :", accuracy_score(all_y_true, y_pred))
print_full_report(all_y_true, y_pred, list(label_map.keys()))

