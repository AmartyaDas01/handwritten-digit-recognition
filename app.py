"""Gradio demo: draw one or more digits, get a live prediction from an MNIST-trained CNN."""

import os
from typing import cast

import numpy as np
import gradio as gr
from PIL import Image
from scipy import ndimage
from tensorflow import keras  # type: ignore[reportMissingModuleSource]

model = keras.models.load_model("model.keras")

MIN_BLOB_AREA = 20  # ignore stray dots/clicks smaller than this many pixels


def to_ink(editor_value):
    """Turn a Gradio Sketchpad value into a white-ink-on-black float32 array.

    The sketchpad gives dark strokes on a light background, so this inverts it to
    match MNIST's white-on-black convention. Preprocessing below (crop, rescale,
    center) is applied per digit, not here.
    """
    if editor_value is None:
        return None

    composite = editor_value.get("composite")
    if composite is None:
        composite = editor_value.get("background")
    if composite is None:
        return None

    img = Image.fromarray(composite).convert("L")
    arr = np.array(img).astype("float32")
    return 255.0 - arr


def _should_merge(a, b):
    """True if two bounding boxes likely belong to the same digit.

    Catches digits like "4" or "5" that a user draws as more than one stroke: boxes
    that overlap horizontally (stacked strokes) always merge; boxes with only a
    small gap relative to their height merge too. Distinct digits, drawn with a
    clear gap, stay separate.
    """
    ax0, _, ax1, ay1 = a
    bx0, _, bx1, by1 = b
    x_gap = bx0 - ax1
    height = max(ay1 - a[1], by1 - b[1], 1)
    return x_gap < min(max(6, 0.25 * height), 16)


def _merge_boxes(boxes):
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = []
    for box in boxes:
        if merged and _should_merge(merged[-1], box):
            px0, py0, px1, py1 = merged[-1]
            x0, y0, x1, y1 = box
            merged[-1] = (min(px0, x0), min(py0, y0), max(px1, x1), max(py1, y1))
        else:
            merged.append(box)
    return merged


def segment_digits(ink):
    """Split an ink array into per-digit bounding boxes, sorted left to right.

    Finds 8-connected ink blobs, then merges blobs that likely belong to the same
    multi-stroke digit. Digits still need a clear gap between them to be told apart.
    """
    mask = ink > 20
    if not mask.any():
        return []

    # scipy ships no type stubs for ndimage.label, so pyright can't infer its
    # (ndarray, int) return; the cast documents what it actually returns.
    labeled, n = cast(
        "tuple[np.ndarray, int]", ndimage.label(mask, structure=np.ones((3, 3)))
    )
    boxes = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if ys.size < MIN_BLOB_AREA:
            continue
        boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))

    return _merge_boxes(boxes)


def format_digit(cropped):
    """Fit one cropped ink region into a (28, 28) array matching MNIST's format:
    scaled to a 20x20 box, then re-centered by center of mass.
    """
    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    resized = np.array(
        Image.fromarray(cropped.astype("uint8")).resize(
            (new_w, new_h), Image.Resampling.LANCZOS
        )
    ).astype("float32")

    canvas = np.zeros((28, 28), dtype="float32")
    top_pad = (28 - new_h) // 2
    left_pad = (28 - new_w) // 2
    canvas[top_pad : top_pad + new_h, left_pad : left_pad + new_w] = resized

    total = canvas.sum()
    if total > 0:
        ys, xs = np.indices(canvas.shape)
        cy = (ys * canvas).sum() / total
        cx = (xs * canvas).sum() / total
        canvas = np.roll(canvas, int(round(14 - cy)), axis=0)
        canvas = np.roll(canvas, int(round(14 - cx)), axis=1)

    return canvas / 255.0


def predict(editor_value):
    ink = to_ink(editor_value)
    if ink is None:
        return ""

    boxes = segment_digits(ink)
    if not boxes:
        return ""

    batch = np.stack(
        [format_digit(ink[y0 : y1 + 1, x0 : x1 + 1]) for x0, y0, x1, y1 in boxes]
    ).reshape(-1, 28, 28, 1)

    preds = model.predict(batch, verbose=0).argmax(axis=1)
    return "".join(str(d) for d in preds)


CSS = "#prediction textarea { font-size: 3rem; text-align: center; }"

with gr.Blocks(title="Handwritten Digit Recognition") as demo:
    gr.Markdown(
        "# Handwritten Digit Recognition\n"
        "Draw one or more digits, **large and with a gap between each one**. "
        "A small CNN trained on MNIST predicts them live."
    )
    with gr.Row():
        sketchpad = gr.Sketchpad(
            label="Draw here",
            image_mode="L",
            type="numpy",
            canvas_size=(280, 280),
            brush=gr.Brush(default_size=18, colors=["#000000"], color_mode="fixed"),
        )
        prediction = gr.Textbox(label="Prediction", elem_id="prediction", scale=1)

    sketchpad.change(predict, inputs=sketchpad, outputs=prediction)

    gr.Examples(
        examples=[
            ["examples/7.png"],
            ["examples/42.png"],
            ["examples/319.png"],
            ["examples/8.png"],
        ],
        inputs=sketchpad,
        outputs=prediction,
        fn=predict,
        cache_examples=False,
        label="Or try an example",
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        css=CSS,
    )
