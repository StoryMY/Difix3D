"""Save training tensors as labeled, side-by-side PNG comparisons."""

from pathlib import Path

import torch
from PIL import Image, ImageDraw


def save_comparisons(source, target, prediction, directory, start_index=0):
    """Inputs are [batch, views, RGB, height, width] tensors in [-1, 1]."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(source.shape[0]):
        reference = source[index, 1] if source.shape[1] > 1 else None
        panels = [source[index, 0], reference, target[index, 0], prediction[index, 0]]
        height, width = source.shape[-2:]
        header = 32
        canvas = Image.new("RGB", (4 * width, height + header), (32, 32, 32))
        draw = ImageDraw.Draw(canvas)
        labels = ["Input", "Reference" if reference is not None else "Ref: N/A", "Target", "Output"]
        for column, (label, tensor) in enumerate(zip(labels, panels)):
            if tensor is not None:
                pixels = tensor.detach().float().cpu().clamp(-1, 1)
                pixels = ((pixels + 1) * 127.5).round().to(dtype=torch.uint8)
                image = Image.fromarray(pixels.permute(1, 2, 0).numpy())
                canvas.paste(image, (column * width, header))
            draw.text((column * width + 4, 8), label, fill="white")
        canvas.save(directory / f"sample_{start_index + index:05d}.png")
