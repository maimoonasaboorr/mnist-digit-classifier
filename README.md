# Handwritten Digit Classifier (MNIST) with TensorFlow/Keras

A simple feedforward neural network that classifies handwritten digits (0-9) from the MNIST dataset, built with `tf.keras.Sequential`.

## What it does

1. Loads the MNIST dataset (60,000 training images, 10,000 test images, each 28x28 grayscale).
2. Builds a fully-connected (dense) network:
   - `Flatten` — reshapes each 28x28 image into a 784-length vector
   - `Dense(128, activation='relu')` — hidden layer
   - `Dense(10, activation='softmax')` — output layer, one probability per digit class
3. Compiles with the Adam optimizer and sparse categorical cross-entropy loss (labels are plain integers, not one-hot).
4. Trains for 5 epochs.
5. Evaluates on the held-out test set and demonstrates a manual single-image prediction.

## Results

| Metric | Value |
|---|---|
| Final training accuracy (epoch 5) | ~94.3% |
| Test accuracy | ~93.8% |

## Requirements

```
tensorflow
numpy
matplotlib
```

Install with:
```bash
pip install tensorflow numpy matplotlib
```

## Usage

Open `NNwithTensorflow.ipynb` in Jupyter or Google Colab and run all cells top to bottom. No external data files are needed — MNIST is downloaded automatically via `tf.keras.datasets.mnist.load_data()`.

## Notebook structure

- **Pre Requisites** — installs/imports
- **Load Dataset** — loads MNIST, inspects shapes, visualizes a sample digit
- **Create Model** — defines the architecture
- **Compile Model** — sets optimizer, loss, metrics
- **Train Model** — fits the model for 5 epochs
- **Evaluate Model** — reports test loss/accuracy
- **Manual testing** — predicts on a single test image and compares to the true label

## Notes

- This is a plain Dense-only network (no convolutional layers), so it treats the image as a flat vector rather than exploiting spatial structure — a reasonable baseline, but a CNN would generally do better on this task.
- Training/test accuracy are close, suggesting minimal overfitting at 5 epochs.
