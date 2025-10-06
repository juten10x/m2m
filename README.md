# 🎵 Motion2Music Local Pipeline

このリポジトリでは、**動画データから骨格座標を抽出 → 潜在変数を生成** するまでを  
ローカル環境で一気に実行できます。

最終的には以下のような流れになります。

```text
動画データ/sample.mp4
  ↓
input/input.csv        ← 99次元の骨格CSV
  ↓
latent/latent_series.csv ← 8次元の潜在変数CSV
```

 1. 仮想環境の作成
まず、このリポジトリをクローンしてフォルダに入ります。

```
git clone https://github.com/sora1871/m2m.git
```
```
python -m venv venv
Windows の場合
venv\Scripts\activate

Mac / Linux の場合
source venv/bin/activate
```
📦 2. 必要なライブラリをインストール
仮想環境を有効化した状態で以下を実行します。

```
pip install -r requirements.txt
```
これで、必要なライブラリ（FastAPI、Torch、Mediapipe、OpenCVなど）が一式インストールされます。

⚙️ 3. サーバーを起動
次のコマンドでローカルサーバーを立てます。

```
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

ブラウザで以下を開くと、APIの確認画面（Swagger UI）が表示されます。
👉 http://127.0.0.1:8000/docs

💡 このサーバーは「99次元CSV → 8次元潜在CSV」を変換するAPIです。

🎬 4. 動画 → 99次元CSV に変換
別のターミナル（またはコマンドプロンプト）を開いて、
動画から骨格データを抽出します。

```
python video_to_pose_csv.py --video 動画データ/sample.mp4 --out input/input.csv
これで以下のファイルが生成されます。

text
コードをコピーする
input/input.csv        ← 99列（33点 × x, y, z）の骨格座標
input/input.meta.json  ← メタ情報（FPSなど）
```

🔮 5. 99次元CSV → 8次元潜在CSV に変換
サーバーを起動したままの状態で、
もう一つターミナルを開き、次のコマンドを実行します。

```
python get_latents.py --in input/input.csv --out latent/latent_series.csv 
```
latent/latent_series.csv  ← 8次元の潜在変数（z）

フォルダ構成（最終形）
```
m2m/
├── .venv/
├── app.py                  # FastAPI サーバー（ローカル推論API）
├── get_latents.py          # 99次元→8次元変換クライアント
├── video_to_pose_csv.py    # 動画→99次元CSV変換スクリプト
├── requirements.txt
├── 動画データ/
│   └── sample.mp4
├── input/
│   ├── input.csv
│   └── input.meta.json
└── latent/
    └── latent_series.csv
```

```
🌟 全体の流れ（図）
text
コードをコピーする
┌────────────────────────────┐
│        動画データ          │
│   (sample.mp4など)         │
└────────────┬─────────────┘
             │
             ▼
┌────────────────────────────┐
│ video_to_pose_csv.py        │
│ → 33点×(x,y,z)=99列CSV出力 │
└────────────┬─────────────┘
             │
             ▼
┌────────────────────────────┐
│  get_latents.py（API経由） │
│ → LSTMモデルで潜在変数生成 │
└────────────┬─────────────┘
             │
             ▼
┌────────────────────────────┐
│   latent/latent_series.csv  │
│     （8次元潜在ベクトル）   │
└────────────────────────────┘
```
