import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frames from a video into an image folder for VGGT-SLAM."
    )
    parser.add_argument("video", type=Path, help="Path to input video file.")
    parser.add_argument("out_dir", type=Path, help="Output directory for extracted frames.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second to extract.")
    parser.add_argument(
        "--ext", type=str, default="jpg", choices=["jpg", "png"], help="Image format."
    )
    parser.add_argument(
        "--start_number", type=int, default=0, help="Starting index for frame numbering."
    )

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = args.out_dir / f"frame_%04d.{args.ext}"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(args.video),
        "-vf",
        f"fps={args.fps}",
        "-start_number",
        str(args.start_number),
        str(out_pattern),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
