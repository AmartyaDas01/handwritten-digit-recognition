# Handwritten Digit Recognition

Draw a digit in the browser with your mouse and a small CNN tells you what it is, live,
with a confidence score for every class.

**Live demo:** _(pending deployment)_

<!-- ![demo](demo.gif) -->

## How it works

A small CNN (two Conv2D + MaxPool blocks, dense head, dropout) is trained on MNIST and
reaches **98.8% test accuracy**. The tricky part of a demo like this isn't the model —
it's making a messy mouse drawing look like an MNIST digit before it hits the model:
the canvas image is inverted to white-on-black, cropped to the drawing's bounding box,
rescaled to fit a 20x20 box, and re-centered by center of mass inside a 28x28 frame,
matching MNIST's own preprocessing.

## Stack

- TensorFlow / Keras
- Gradio (Sketchpad input)
- Hugging Face Spaces for hosting

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py   # trains the CNN and saves model.keras
python app.py      # launches the Gradio app at http://127.0.0.1:7860
```
