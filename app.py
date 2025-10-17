# app.py
# ローカル実行向け FastAPI サーバ（本番の api/main.py をベースに簡素化）
# 起動: uvicorn app:app --host 127.0.0.1 --port 8000 --reload

import os
import io
import json
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Response
from pydantic import BaseModel
import joblib

# CORSMiddleware（Access-Control-Allow-Origin ヘッダーを設定する用）
from fastapi.middleware.cors import CORSMiddleware

# ========== 設定（ローカル既定・環境変数で上書き可） ==========
MODEL_PATH = os.environ.get("MODEL_PATH", "models/lstm_0814_model.pt")
SCALER_PATH = os.environ.get("SCALER_PATH", "models/scaler.joblib")
FEATURE_COLS_PATH = os.environ.get("FEATURE_COLS_PATH", "models/feature_cols.json")  # 任意

INPUT_DIM = int(os.environ.get("INPUT_DIM", "99"))
LATENT_DIM = int(os.environ.get("LATENT_DIM", "8"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "128"))

API_KEY_REQ = os.environ.get("API_KEY", None)  # 無指定なら API キー不要
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========== ユーティリティ ==========
def create_windows(data: np.ndarray, window_size: int = 20, stride: int = 1) -> np.ndarray:
    wins = []
    for i in range(0, len(data) - window_size + 1, stride):
        wins.append(data[i:i + window_size])
    return np.array(wins)


class LSTMAutoEncoder(nn.Module):
    """Encoder で潜在 z を作り、Decoder で再構成する学習時の構造を再現"""
    def __init__(self, input_dim=99, hidden_dim=128, latent_dim=8):
        super().__init__()
        # Encoder
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Encode
        _, (h_n, _) = self.encoder_lstm(x)   # h_n: [1, B, H]
        h_last = h_n[-1]                      # [B, H]
        z = self.to_latent(h_last)            # [B, Z]
        # Decode（再構成は返すが、APIは z を利用）
        h_dec = self.from_latent(z)           # [B, H]
        h_dec_seq = h_dec.unsqueeze(1).repeat(1, x.size(1), 1)  # [B, T, H]
        dec_out, _ = self.decoder_lstm(h_dec_seq)               # [B, T, H]
        recon = self.output_layer(dec_out)                      # [B, T, D]
        return recon, z


def _ensure_exists(path: str, name: str):
    if not os.path.exists(path):
        raise RuntimeError(f"{name} not found: {path}")


def load_feature_cols(path: str) -> Optional[List[str]]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def pick_numeric_in_order(df: pd.DataFrame, expected_dim: int, feature_cols: Optional[List[str]]) -> pd.DataFrame:
    """学習時の列順があれば強制。無ければ数値列のみ。次元不一致なら 400。"""
    if feature_cols:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise HTTPException(400, f"入力CSVに必要な列が不足: {missing[:5]} ...")
        df_num = df[feature_cols]
    else:
        df_num = df.select_dtypes(include="number")
    if df_num.shape[1] != expected_dim:
        raise HTTPException(
            400,
            f"期待次元={expected_dim} ですが、入力は {df_num.shape[1]} 列です。列順・余計な数値列を確認してください。"
        )
    return df_num


def windows_to_scaled_tensor(X: np.ndarray, window_size: int, stride: int, scaler) -> torch.Tensor:
    """[L,D] -> window化->[N,T,D] -> flatten -> scaler.transform -> reshape -> tensor(float32)"""
    wins = create_windows(X, window_size, stride)  # [N,T,D]
    if wins.shape[0] == 0:
        return torch.empty((0, window_size, X.shape[1]), dtype=torch.float32, device=DEVICE)
    flat = wins.reshape(-1, X.shape[1])            # [N*T, D]
    scaled = scaler.transform(flat).reshape(wins.shape).astype(np.float32)
    return torch.from_numpy(scaled).to(DEVICE)


def verify_api_key(x_api_key: Optional[str]) -> None:
    if API_KEY_REQ and x_api_key != API_KEY_REQ:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ========== 起動時ロード ==========
_ensure_exists(MODEL_PATH, "MODEL_PATH")
_ensure_exists(SCALER_PATH, "SCALER_PATH")

scaler = joblib.load(SCALER_PATH)
feature_cols = load_feature_cols(FEATURE_COLS_PATH)

model = LSTMAutoEncoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM)
state = torch.load(MODEL_PATH, map_location=DEVICE)

# いくつかの保存形式に対応
if isinstance(state, dict) and "state_dict" in state and not any(k.startswith("encoder_lstm") for k in state.keys()):
    state = state["state_dict"]
elif isinstance(state, dict) and "model_state_dict" in state and not any(k.startswith("encoder_lstm") for k in state.keys()):
    state = state["model_state_dict"]

model.load_state_dict(state, strict=True)
model.to(DEVICE).eval()


# ========== FastAPI 定義 ==========
app = FastAPI(title="LSTM-AE Latent API (local)", version="1.0.0")


class ArrayRequest(BaseModel):
    data: List[List[float]]    # [L, D]
    window_size: int = 20
    stride: int = 1
    return_meta: bool = False


class ArrayResponse(BaseModel):
    n_windows: int
    latent_dim: int
    z: List[List[float]]
    starts: Optional[List[int]] = None


# Access-Control-Allow-Origin ヘッダーを設定
origins = [
    "https://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "input_dim": INPUT_DIM,
        "latent_dim": LATENT_DIM,
        "has_feature_cols": bool(feature_cols),
        "model_path": MODEL_PATH,
        "scaler_path": SCALER_PATH,
    }


@app.head("/health")
def health_head():
    return Response(status_code=200)


@app.post("/v1/latents/from-array", response_model=ArrayResponse)
def latents_from_array(payload: ArrayRequest, x_api_key: Optional[str] = Header(default=None)):
    verify_api_key(x_api_key)

    X = np.asarray(payload.data, dtype=np.float32)  # [L, D]
    if X.ndim != 2 or X.shape[1] != INPUT_DIM:
        raise HTTPException(400, f"data の形が不正です。期待 [L,{INPUT_DIM}] ですが {list(X.shape)} を受け取りました。")

    x_t = windows_to_scaled_tensor(X, payload.window_size, payload.stride, scaler)  # [N,T,D]
    if x_t.shape[0] == 0:
        return ArrayResponse(n_windows=0, latent_dim=LATENT_DIM, z=[], starts=[] if payload.return_meta else None)

    with torch.no_grad():
        _, z = model(x_t)                   # [N, Z]
        z_np = z.detach().cpu().numpy().tolist()

    starts = list(range(0, X.shape[0] - payload.window_size + 1, payload.stride)) if payload.return_meta else None
    return ArrayResponse(n_windows=len(z_np), latent_dim=LATENT_DIM, z=z_np, starts=starts)


@app.post("/v1/latents/from-csv", response_model=ArrayResponse)
async def latents_from_csv(file: UploadFile = File(...),
                           window_size: int = 20,
                           stride: int = 1,
                           return_meta: bool = False,
                           x_api_key: Optional[str] = Header(default=None)):
    verify_api_key(x_api_key)

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(400, "CSV の読込に失敗しました")

    df_num = pick_numeric_in_order(df, expected_dim=INPUT_DIM, feature_cols=feature_cols)
    X = df_num.to_numpy(dtype=np.float32)  # [L, D]

    x_t = windows_to_scaled_tensor(X, window_size, stride, scaler)  # [N,T,D]
    if x_t.shape[0] == 0:
        return ArrayResponse(n_windows=0, latent_dim=LATENT_DIM, z=[], starts=[] if return_meta else None)

    with torch.no_grad():
        _, z = model(x_t)
        z_np = z.detach().cpu().numpy().tolist()

    starts = list(range(0, X.shape[0] - window_size + 1, stride)) if return_meta else None
    return ArrayResponse(n_windows=len(z_np), latent_dim=LATENT_DIM, z=z_np, starts=starts)
