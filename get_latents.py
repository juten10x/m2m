import requests, pandas as pd

BASE_URL = "http://127.0.0.1:8000"
url = f"{BASE_URL}/v1/latents/from-csv"
params = {"window_size": 20, "stride": 1, "return_meta": True}

with open("input/input.csv", "rb") as f:
    res = requests.post(url, files={"file": ("input.csv", f, "text/csv")}, params=params, timeout=120)
res.raise_for_status()
data = res.json()

df = pd.DataFrame(data["z"])
if data.get("starts") is not None:
    df.insert(0, "start_frame", data["starts"])
df.to_csv("latent/latent_series.csv", index=False)
print("保存しました: latent_series.csv")
