# run_pipeline.py
import argparse, subprocess, sys

def run(cmd):
    print(">", " ".join(cmd)); rc = subprocess.call(cmd)
    if rc != 0: sys.exit(rc)

def main():
    ap = argparse.ArgumentParser(description="video -> pose CSV -> latent CSV")
    ap.add_argument("--video", required=True)
    ap.add_argument("--pose_csv", default="input/input.csv")
    ap.add_argument("--latent_csv", default="latent/latent_series.csv")
    ap.add_argument("--base_url", default="http://127.0.0.1:8000")
    ap.add_argument("--window_size", type=int, default=20)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--normalize", choices=["none","hip"], default="none")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--api_key", default=None)
    args = ap.parse_args()

    # 1) 動画→CSV
    cmd1 = ["python","video_to_pose_csv.py","--video",args.video,"--out",args.pose_csv,
            "--normalize",args.normalize]
    if args.fps: cmd1 += ["--fps", str(args.fps)]
    run(cmd1)

    # 2) CSV→潜在CSV
    cmd2 = ["python","get_latents.py","--in",args.pose_csv,"--out",args.latent_csv,
            "--base_url",args.base_url,"--window_size",str(args.window_size),
            "--stride",str(args.stride)]
    if args.api_key: cmd2 += ["--api_key", args.api_key]
    run(cmd2)

    print("🎉 完了:", args.latent_csv)

if __name__ == "__main__":
    main()
