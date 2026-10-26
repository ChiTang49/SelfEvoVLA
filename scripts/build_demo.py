import argparse
import json
import subprocess
from pathlib import Path


METHODS = {
    "sft": ("E7-paired-video-sft", "SFT"),
    "rloo": ("E7-paired-video-rloo", "RLOO"),
    "e3": ("E7-paired-video-e3", "Bounded RIPT"),
}


def load_records(root):
    grouped = {}
    flat = []
    for method, (experiment, label) in METHODS.items():
        path = root / "evidence" / experiment / "episode_results.json"
        payload = json.loads(path.read_text())
        for episode in payload["episodes"]:
            index = episode["init_state_index"]
            video = root / "evidence" / experiment / "videos" / Path(episode["video_path"]).name
            if not video.is_file():
                raise FileNotFoundError(video)
            record = {
                "method": method,
                "method_label": label,
                "experiment_id": experiment,
                "init_state_index": index,
                "success": episode["success"],
                "episode_steps": episode["episode_steps"],
                "video": str(video),
            }
            grouped.setdefault(index, {})[method] = record
            flat.append({key: value for key, value in record.items() if key != "video"})
    if len(flat) != 24 or any(set(records) != set(METHODS) for records in grouped.values()):
        raise ValueError("Expected eight paired initial states for all three methods")
    return grouped, flat


def select_cases(grouped):
    common_failure = next((index for index, records in grouped.items() if not any(r["success"] for r in records.values())), None)
    divergence = next((index for index, records in grouped.items() if records["e3"]["success"] and not records["sft"]["success"]), None)
    if divergence is None:
        divergence = next((index for index, records in grouped.items() if len({r["success"] for r in records.values()}) > 1), None)
    remaining = [index for index in sorted(grouped) if index not in {common_failure, divergence}]
    cases = [index for index in (common_failure, divergence) if index is not None]
    cases.extend(remaining[: 3 - len(cases)])
    if len(cases) != 3:
        raise ValueError("Could not select three distinct paired cases")
    return cases


def run(command):
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--cases", help="Comma-separated initial-state indices; default uses the declared selection rule")
    parser.add_argument("--seconds-per-case", type=int, default=25)
    parser.add_argument("--font", default="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    args = parser.parse_args()

    grouped, flat = load_records(args.root)
    cases = [int(value) for value in args.cases.split(",")] if args.cases else select_cases(grouped)
    if len(cases) != 3 or len(set(cases)) != 3:
        raise ValueError("Exactly three distinct cases are required")

    assets = args.root / "assets"
    assets.mkdir(exist_ok=True)
    selection = {
        "selection_rule": "Prefer one common failure, one E3-success/SFT-failure divergence, then the lowest remaining index; all eight paired groups are reported.",
        "selected_init_state_indices": cases,
        "paired_episodes": flat,
    }
    (assets / "demo_manifest.json").write_text(json.dumps(selection, indent=2) + "\n")

    segments = []
    for segment_index, init_index in enumerate(cases):
        command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y"]
        for method in METHODS:
            command.extend(["-i", grouped[init_index][method]["video"]])
        filters = []
        for input_index, method in enumerate(METHODS):
            record = grouped[init_index][method]
            filters.append(
                f"[{input_index}:v]fps=24,scale=480:360:force_original_aspect_ratio=decrease,"
                f"pad=480:360:(ow-iw)/2:(oh-ih)/2:color=0x101827,"
                f"trim=duration={args.seconds_per_case},setpts=PTS-STARTPTS,"
                f"tpad=stop_mode=clone:stop_duration={args.seconds_per_case},"
                f"trim=duration={args.seconds_per_case},"
                f"drawtext=fontfile={args.font}:text='{record['episode_steps']} env steps':"
                "fontcolor=white:fontsize=14:x=12:y=51:box=1:boxcolor=0x101827@0.75:boxborderw=5[v{0}]".format(input_index)
            )
        filters.append("[v0][v1][v2]hstack=inputs=3[combined]")
        for input_index, method in enumerate(METHODS):
            record = grouped[init_index][method]
            status = "SUCCESS" if record["success"] else "FAIL"
            color = "0x16805b" if record["success"] else "0xb33a3a"
            source = "combined" if input_index == 0 else f"header{input_index - 1}"
            filters.append(
                f"[{source}]drawtext=fontfile={args.font}:text='{record['method_label']} | {status}':"
                f"fontcolor=white:fontsize=19:x={12 + input_index * 480}:y=10:"
                f"box=1:boxcolor={color}@0.95:boxborderw=7[header{input_index}]"
            )
        filters.append(
            f"[header2]drawbox=x=0:y=ih-48:w=iw:h=48:color=0x101827:t=fill,"
            f"drawtext=fontfile={args.font}:text='Init state {init_index} | Put alphabet soup and tomato sauce in the basket':"
            "fontcolor=white:fontsize=20:x=18:y=h-33[out]"
        )
        segment = assets / f".demo_segment_{segment_index}.mp4"
        command.extend([
            "-filter_complex", ";".join(filters), "-map", "[out]", "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(segment),
        ])
        run(command)
        segments.append(segment)

    concat_file = assets / ".demo_concat.txt"
    concat_file.write_text("".join(f"file '{segment.name}'\n" for segment in segments))
    output = assets / "ript_comparison_demo.mp4"
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_file), "-c", "copy", "-movflags", "+faststart", str(output),
    ])
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-ss", "1", "-i", str(output),
        "-frames:v", "1", str(assets / "demo_cover.png"),
    ])
    for segment in segments:
        segment.unlink()
    concat_file.unlink()
    print(json.dumps({"video": str(output), "selected_cases": cases, "duration_seconds": args.seconds_per_case * 3}))


if __name__ == "__main__":
    main()
