import argparse
import sys
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd

# mediapipe は v0.10+ で mp.solutions の構造が変わりましたが、
# 下記 import は 0.8～0.10 系で動く一般的な形です
import mediapipe as mp

# Mediapipe Pose のランドマーク数（顔含む全身）：33
NUM_POINTS = 33
AXES = ("x", "y", "z")
COLUMNS_99 = [f"P{i}_{ax}" for i in range(NUM_POINTS) for ax in AXES]

# 腰まわりの基準点（左右の臀部）
LEFT_HIP  = 23
RIGHT_HIP = 24
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12

def extract_pose_from_video(
    video_path: str,
    model_complexity: int = 1,
    static_image_mode: bool = False,
    smooth_landmarks: bool = True,
) -> pd.DataFrame | None:
    """
    動画から Mediapipe Pose を使って 33 点の (x,y,z) をフレームごとに抽出。
    返り値は shape=(フレーム数, 99) の DataFrame（列名は P{i}_{axis}）。
    検出失敗フレームは NaN 行で埋めて後段の補完に回します。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}", file=sys.stderr)
        return None

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        enable_segmentation=False
    )

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    rows = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            row = []
            for lm in results.pose_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])  # 画面座標系に正規化済み（0-1）。zは相対深度（マイナス手前）
            rows.append(row)
        else:
            # 検出できないフレームは NaN で埋める（後で補完）
            rows.append([np.nan] * (NUM_POINTS * 3))

    cap.release()
    pose.close()

    if not rows:
        print(f"⚠️ No frames read from {video_path}", file=sys.stderr)
        return None

    df = pd.DataFrame(rows, columns=COLUMNS_99)
    # もし総フレーム数を取得できて、読み込んだ数と大きくずれていたら警告（可）
    if frames and abs(len(df) - frames) > max(3, int(0.01 * frames)):
        print(f"ℹ️ Warning: extracted {len(df)} rows but video reports {frames} frames.", file=sys.stderr)
    return df


def hip_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    腰中心化 + スケール正規化（左右肩幅 or 腰幅を 1 に目安）。
    - 各フレームで左右臀部の中点を原点へ移動（平行移動）
    - スケールは肩幅を優先、欠損なら腰幅。極端に小さい時はスキップ
    """
    arr = df.to_numpy(float).reshape(len(df), NUM_POINTS, 3)  # (T, 33, 3)
    # 中心（腰の中点）
    hip_center = (arr[:, LEFT_HIP, :] + arr[:, RIGHT_HIP, :]) / 2.0
    arr = arr - hip_center[:, None, :]  # 平行移動

    # スケーリング（肩幅→腰幅）
    shoulder_w = np.linalg.norm(arr[:, LEFT_SHOULDER, :] - arr[:, RIGHT_SHOULDER, :], axis=1)
    hip_w      = np.linalg.norm(arr[:, LEFT_HIP, :] - arr[:, RIGHT_HIP, :], axis=1)
    scale = np.where(np.isfinite(shoulder_w) & (shoulder_w > 1e-6), shoulder_w, hip_w)
    # 数値安定化
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    arr = arr / scale[:, None, None]

    out = arr.reshape(len(df), NUM_POINTS * 3)
    return pd.DataFrame(out, columns=df.columns, index=df.index)


def interpolate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    欠損補完＆簡易スムージング（お好みで変更可）
    - 線形補完（前後方向）
    - 前後で補えない端は前方/後方埋め
    - 低振幅ノイズ抑制のため移動平均（窓=3）
    """
    out = df.copy()
    out.interpolate(method="linear", limit_direction="both", inplace=True)
    out.fillna(method="ffill", inplace=True)
    out.fillna(method="bfill", inplace=True)
    # 簡易スムージング（任意）
    out = out.rolling(window=3, min_periods=1, center=True).mean()
    return out


def resample_to_fps(df: pd.DataFrame, src_fps: float | None, target_fps: float | None) -> pd.DataFrame:
    """
    等間隔サンプリングでFPS合わせ（src_fps が不明でもフレーム数から近似）。
    target_fps 未指定ならそのまま返す。
    """
    if not target_fps:
        return df
    if src_fps and src_fps > 0:
        ratio = target_fps / src_fps
    else:
        # fps不明時は近似：長さを ratio 倍に
        ratio = target_fps / 30.0  # 仮に30fps基準
    if abs(ratio - 1.0) < 1e-3:
        return df

    new_len = max(1, int(round(len(df) * ratio)))
    # 線形補間（時間軸正規化）
    old_idx = np.linspace(0, 1, len(df))
    new_idx = np.linspace(0, 1, new_len)
    out = pd.DataFrame(index=range(new_len), columns=df.columns, dtype=float)
    for c in df.columns:
        out[c] = np.interp(new_idx, old_idx, df[c].astype(float).to_numpy())
    return out


def read_fps(video_path: str) -> float | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps and np.isfinite(fps) and fps > 0:
        return float(fps)
    return None


def main():
    ap = argparse.ArgumentParser(description="Extract pose (x,y,z) from video and save as CSV.")
    ap.add_argument("--video", default="動画データ/sample.mp4", help="Input video path")
    ap.add_argument("--out", default="input/input.csv", help="Output CSV path")
    ap.add_argument("--normalize", choices=["none", "hip"], default="none")
    ap.add_argument("--fps", type=float, default=None)
    args = ap.parse_args()

    video_path = args.video
    out_csv = args.out

    print(f"▶ Extracting pose from: {video_path}")
    df = extract_pose_from_video(video_path)
    if df is None or df.empty:
        print("❌ No landmarks extracted.")
        sys.exit(1)

    df = interpolate_and_clean(df)
    if args.normalize == "hip":
        df = hip_normalize(df)
    src_fps = read_fps(video_path)
    if args.fps:
        df = resample_to_fps(df, src_fps, args.fps)

    # 出力ディレクトリを作成
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ Saved: {out_csv}  (frames={len(df)}, dims={df.shape[1]})")

    meta_path = Path(out_csv).with_suffix(".meta.json")
    meta = {
        "source_video": str(video_path),
        "frames": len(df),
        "input_dim": df.shape[1],
        "fps_source": src_fps,
        "fps_target": args.fps,
        "normalized": args.normalize,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"ℹ️ Meta saved: {meta_path}")


if __name__ == "__main__":
    main()