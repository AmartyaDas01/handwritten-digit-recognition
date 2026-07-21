"""Gradio demo: draw a digit, get a prediction from an MNIST-trained CNN."""

import numpy as np
import gradio as gr
from PIL import Image
from tensorflow import keras

model = keras.models.load_model("model.keras")


def preprocess(editor_value):
    """Turn a Gradio Sketchpad value into a (1, 28, 28, 1) array matching MNIST's format.

    MNIST digits are white strokes on black, centered by center of mass with the
    digit scaled to fit a 20x20 box inside the 28x28 frame. The sketchpad gives dark
    strokes on a light background, so this inverts, crops to the drawing's bounding
    box, rescales, and re-centers to match.
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

    # Sketchpad is dark strokes on a light background -> invert to white-on-black.
    arr = 255.0 - arr

    # Drop near-empty pixels, then crop to the drawing's bounding box.
    mask = arr > 20
    if not mask.any():
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    cropped = arr[top : bottom + 1, left : right + 1]

    # Resize so the longest side is 20px, preserving aspect ratio.
    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    resized = np.array(
        Image.fromarray(cropped.astype("uint8")).resize((new_w, new_h), Image.LANCZOS)
    ).astype("float32")

    # Paste centered into a 28x28 canvas (~4px padding on the longest side).
    canvas = np.zeros((28, 28), dtype="float32")
    top_pad = (28 - new_h) // 2
    left_pad = (28 - new_w) // 2
    canvas[top_pad : top_pad + new_h, left_pad : left_pad + new_w] = resized

    # Re-center by center of mass, as in the original MNIST preprocessing.
    total = canvas.sum()
    if total > 0:
        ys, xs = np.indices(canvas.shape)
        cy = (ys * canvas).sum() / total
        cx = (xs * canvas).sum() / total
        shift_y = int(round(14 - cy))
        shift_x = int(round(14 - cx))
        canvas = np.roll(canvas, shift_y, axis=0)
        canvas = np.roll(canvas, shift_x, axis=1)

    canvas = canvas / 255.0
    return canvas.reshape(1, 28, 28, 1)


def predict(editor_value):
    x = preprocess(editor_value)
    if x is None:
        return {str(i): 0.0 for i in range(10)}
    probs = model.predict(x, verbose=0)[0]
    return {str(i): float(probs[i]) for i in range(10)}


with gr.Blocks(title="Handwritten Digit Recognition") as demo:
    gr.Markdown(
        "# Handwritten Digit Recognition\n"
        "Draw a single digit (0-9), **large and centered** in the box. "
        "A small CNN trained on MNIST predicts it live."
    )
    with gr.Row():
        sketchpad = gr.Sketchpad(
            label="Draw here",
            image_mode="L",
            type="numpy",
            canvas_size=(280, 280),
            brush=gr.Brush(default_size=18, colors=["#000000"], color_mode="fixed"),
        )
        label = gr.Label(label="Prediction", num_top_classes=10)

    sketchpad.change(predict, inputs=sketchpad, outputs=label)

if __name__ == "__main__":
    demo.launch()
