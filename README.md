# BlidScribe

Draw a word or string of characters on a canvas → the backend segments it into
individual characters, classifies each one with a CNN trained on EMNIST, and
reassembles the predicted string — with per-character confidence and
Grad-CAM explainability.

This is an end-to-end project: dataset → trained model → ONNX export →
FastAPI inference service → browser canvas client. Unlike a plain MNIST demo,
the interesting engineering problem here is **segmentation and train/serve
consistency**, not just classification accuracy.
