import os
from pathlib import Path

import numpy as np
import librosa
from tqdm import tqdm

# ---------------------------
# 1) 개별 파일 -> Mel-Spectrogram 추출 함수
# ---------------------------
def extract_mel(
    file_path: str,
    sr: int = None,           # 원하는 샘플링레이트 (None이면 원본 sr 사용)
    n_mels: int = 128,        # Mel 필터 수
    n_fft: int = 1024,        # FFT 윈도 길이(≈ 46 ms @22.05 kHz)
    hop_length: int = 512,    # Hop 길이(≈ 23 ms @22.05 kHz)
    fmin: int = 0,            # 최소 주파수
    fmax: int = None,         # 최대 주파수(None이면 sr/2)
    power: float = 2.0,       # 1.0(에너지), 2.0(파워)
    to_db: bool = True,       # dB 스케일로 변환할지 여부
) -> np.ndarray:
    """하나의 오디오 파일을 Mel-Spectrogram (frames × n_mels)로 변환"""
    # ① 로드
    y, sr_loaded = librosa.load(file_path, sr=sr, mono=True)

    # ② Mel-Spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr_loaded,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax or sr_loaded // 2,
        power=power,
    )
    
    # (선택) dB 스케일로 변환
    if to_db:
        mel = librosa.power_to_db(mel, ref=np.max)
        
    return mel

# -------------------------------------
# 2) 폴더 안 모든 음성 파일 일괄 변환
# -------------------------------------
def batch_extract_mel(
    input_dir: str,
    output_dir: str,
    pattern: tuple = (".wav", ".flac", ".mp3"),
    **kwargs
):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in input_dir.rglob("*") if p.suffix.lower() in pattern]
    failed_files = []

    for p in tqdm(files, desc="Mel-Spec 변환"):
        try:
            mel = extract_mel(str(p), **kwargs)
            np.save(output_dir / f"{p.stem}.npy", mel)
        except Exception as e:
            print(f"\n[오류] 파일 처리 실패: {p.name} | 원인: {e}")
            failed_files.append(p.name)

    if failed_files:
        print("\n--- 다음 파일들은 오류로 인해 처리를 건너뛰었습니다 ---")
        for f in failed_files:
            print(f" - {f}")

# ---------------------------
# 3) 스크립트 실행 예시
# ---------------------------
if __name__ == "__main__":
    # --- 여러 클래스(폴더)를 한 번에 처리하도록 개선된 예시 ---

    # 1. 변환할 원본 WAV 파일들이 들어있는 상위 폴더
    BASE_INPUT_DIR = "converted_wavs"

    # 2. 변환된 Mel-Spectrogram(.npy)을 저장할 상위 폴더
    BASE_OUTPUT_DIR = "mel_specs 3"

    # 3. BASE_INPUT_DIR 안의 모든 하위 폴더 목록을 자동으로 가져오기
    try:
        class_folders = [d for d in os.listdir(BASE_INPUT_DIR) if os.path.isdir(os.path.join(BASE_INPUT_DIR, d))]
    except FileNotFoundError:
        print(f"[오류] 입력 폴더를 찾을 수 없습니다: '{BASE_INPUT_DIR}'")
        class_folders = []

    if not class_folders:
        print(f"'{BASE_INPUT_DIR}'에서 처리할 하위 폴더를 찾지 못했습니다. 폴더 구조를 확인해주세요.")
    else:
        print(f"총 {len(class_folders)}개의 클래스 폴더를 처리합니다: {class_folders}")

        for class_name in class_folders:
            input_folder = os.path.join(BASE_INPUT_DIR, class_name)
            output_folder = os.path.join(BASE_OUTPUT_DIR, class_name)

            print(f"\n--- 클래스 '{class_name}' 처리 시작 ---")
            batch_extract_mel(
                input_dir=input_folder,
                output_dir=output_folder,
                sr=None,  # 원본 샘플링레이트 사용
                n_mels=128,
                n_fft=1024,
                hop_length=160,
                to_db=True, # dB 스케일로 변환
            )
            print(f"--- 클래스 '{class_name}' 처리 완료 ---")

        print("\n✅ 모든 작업이 성공적으로 완료되었습니다.")
