"""Convert Izanagi-Fix train/val JSONL manifests to a Difix dataset JSON."""

import argparse
import json
from pathlib import Path, PureWindowsPath


def convert_split(manifest, use_reference=True, prompt="remove degradation"):
    manifest = Path(manifest).resolve()
    samples = {}
    with manifest.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            context = f"{manifest}:{line_number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{context}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{context}: expected a JSON object")

            def image_path(field):
                value = row.get(field)
                if field in ("lq", "gt"):
                    value = value or row.get(field.upper())
                if not isinstance(value, str) or not value.strip():
                    hint = "; generate references with --ref-config, or use --no-ref" if field == "ref_image" else ""
                    raise ValueError(f"{context}: missing/non-string {field}{hint}")
                # Keep absolute paths portable; resolve relative paths against the manifest.
                if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
                    return value
                return str((manifest.parent / value).resolve())

            sample = {
                "image": image_path("lq"),
                "target_image": image_path("gt"),
                "prompt": prompt,
            }
            if use_reference:
                sample["ref_image"] = image_path("ref_image")
            samples[f"sample_{line_number:08d}"] = sample
    return samples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Izanagi-Fix train.jsonl")
    parser.add_argument("--val", type=Path, required=True, help="Izanagi-Fix val.jsonl; maps to Difix test")
    parser.add_argument("--output", type=Path, required=True, help="New Difix dataset JSON (must not exist)")
    parser.add_argument("--prompt", default="remove degradation")
    parser.add_argument("--no-ref", action="store_true", help="Omit references for single-view training (do not use --mv_unet)")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}; choose a new --output")
    data = {
        split: convert_split(path, not args.no_ref, args.prompt)
        for split, path in (("train", args.train), ("test", args.val))
    }
    if not data["train"]:
        raise ValueError("Training manifest is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # ASCII escapes also work with the legacy dataset reader's system encoding.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=True, indent=2)
        stream.write("\n")
    print(f"Saved {args.output}: train={len(data['train'])}, test={len(data['test'])}")
    if not data["test"]:
        print("Validation is empty; use --eval_freq 0 when training.")


if __name__ == "__main__":
    main()
