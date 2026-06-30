# Phase IV: Computer Vision Foundations

Phase IV to focus exclusively on how AI handles pixels and visual frames. and we will learn how AI transforms a raw grid of visual color pixels into spatial representations.

## How AI Sees Images: Pixels and Channels

To a computer, an image is not an boject; it's just a multi-dimensional array of integers, usually ranging from 0 (completely black) to 255 (completely white)

- **Grayscale Images (2D Tensors):** A A single grid of numbers representing brightness. Shape: (Height, Width).

- **Color Images (3D Tensors):** Color images stack three separate grids on top of each other—one for Red, Green, and Blue channels. Shape: (Channels, Height, Width) or (3, H, W).

## Preparing Images for your Mini-model (The Multimodal Prep)

Since  we are want to build a model that can "read image" we need to know how to bridge the gap between vision data and a text model

A text model expects a sequence of words: [Word 1, Word 2, Word 3]. To feed an image into that same network, we have to slice the image up into a sequence of small visual squares (called Patches), flatten those patches into vectors, and pass them into the model just like a sentence!
