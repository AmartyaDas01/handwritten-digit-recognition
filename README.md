# Handwritten Digit Recognition

Draw one or more digits in the browser with your mouse and a small CNN reads them
back live — draw a "7", or draw "3", "4", "5" in a row and get "345".

**Live demo:** _(pending deployment)_ — note: the free hosting tier spins down after
15 minutes idle, so the first load after a while can take 30-50s to wake up.

<!-- ![demo](demo.gif) -->

## How it works

A small CNN (two Conv2D + MaxPool blocks, dense head, dropout) is trained on MNIST and
reaches **98.8% test accuracy**. The tricky part of a demo like this isn't the model —
it's making a messy mouse drawing look like an MNIST digit before it hits the model:
the canvas image is inverted to white-on-black, then each digit is found via
connected-component segmentation, cropped to its bounding box, rescaled to fit a 20x20
box, and re-centered by center of mass inside a 28x28 frame, matching MNIST's own
preprocessing. Multiple digits are segmented left to right and classified one at a
time, so draw them with a clear gap between each one.

## Stack

- TensorFlow / Keras
- Gradio (Sketchpad input)
- Hosted on Render (free web service)

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py   # trains the CNN and saves model.keras
python app.py      # launches the Gradio app at http://127.0.0.1:7860
```
