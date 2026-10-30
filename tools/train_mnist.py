#!/usr/bin/env python3
"""Train and export a tiny quantized MNIST linear classifier for VampireGrad."""

from __future__ import annotations

import argparse
import gzip
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

MAGIC = b"VGRD"
FORMAT_VERSION = 1
DTYPE_INT8 = 2
HEADER_SIZE = 16
LAYER_HEADER_SIZE = 8
MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "train-images-idx3-ubyte.gz": (60_000, 28, 28),
    "train-labels-idx1-ubyte.gz": (60_000,),
    "t10k-images-idx3-ubyte.gz": (10_000, 28, 28),
    "t10k-labels-idx1-ubyte.gz": (10_000,),
}


@dataclass(frozen=True)
class Tensor:
    """Minimal host tensor matching VampireGrad's flat row-major representation."""

    shape: tuple[int, ...]
    data: NDArray[np.float32]

    def __post_init__(self) -> None:
        expected = int(np.prod(self.shape, dtype=np.int64))
        if self.data.dtype != np.float32 or self.data.ndim != 1 or self.data.size != expected:
            raise ValueError("tensor shape does not match flat float32 data")

    def matrix(self) -> NDArray[np.float32]:
        return self.data.reshape(self.shape)


def download_mnist(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for filename in MNIST_FILES:
        destination = cache_dir / filename
        if destination.exists():
            continue
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(MNIST_BASE_URL + filename, destination)


def load_images(path: Path, expected_count: int) -> Tensor:
    with gzip.open(path, "rb") as source:
        magic, count, rows, columns = struct.unpack(">IIII", source.read(16))
        if magic != 2051 or count != expected_count or rows != 28 or columns != 28:
            raise ValueError(f"unexpected MNIST image header in {path}")
        pixels = np.frombuffer(source.read(), dtype=np.uint8)
    expected_pixels = count * rows * columns
    if pixels.size != expected_pixels:
        raise ValueError(f"truncated MNIST image file: {path}")
    flattened = pixels.astype(np.float32) / np.float32(255.0)
    return Tensor((count, rows * columns), flattened)


def load_labels(path: Path, expected_count: int) -> NDArray[np.uint8]:
    with gzip.open(path, "rb") as source:
        magic, count = struct.unpack(">II", source.read(8))
        if magic != 2049 or count != expected_count:
            raise ValueError(f"unexpected MNIST label header in {path}")
        labels = np.frombuffer(source.read(), dtype=np.uint8).copy()
    if labels.size != count:
        raise ValueError(f"truncated MNIST label file: {path}")
    return labels


def train(
    images: Tensor,
    labels: NDArray[np.uint8],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    samples, features = images.shape
    classes = 10
    x = images.matrix()
    weights = np.zeros((classes, features), dtype=np.float32)
    bias = np.zeros(classes, dtype=np.float32)
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        order = rng.permutation(samples)
        total_loss = 0.0
        correct = 0
        for start in range(0, samples, batch_size):
            indices = order[start : start + batch_size]
            batch_x = x[indices]
            batch_y = labels[indices]
            logits = batch_x @ weights.T + bias
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)

            total_loss -= float(np.log(probabilities[np.arange(batch_y.size), batch_y] + 1e-12).sum())
            correct += int((probabilities.argmax(axis=1) == batch_y).sum())
            probabilities[np.arange(batch_y.size), batch_y] -= 1.0
            probabilities /= np.float32(batch_y.size)
            weights -= np.float32(learning_rate) * (probabilities.T @ batch_x)
            bias -= np.float32(learning_rate) * probabilities.sum(axis=0)

        print(
            f"epoch {epoch + 1:2d}/{epochs}: "
            f"loss={total_loss / samples:.4f}, accuracy={correct / samples:.2%}"
        )

    return (
        Tensor(weights.shape, weights.reshape(-1)),
        Tensor(bias.shape, bias.reshape(-1)),
    )


def accuracy(images: Tensor, labels: NDArray[np.uint8], weights: Tensor, bias: Tensor) -> float:
    logits = images.matrix() @ weights.matrix().T + bias.matrix()
    return float((logits.argmax(axis=1) == labels).mean())


def quantize(weights: Tensor, bias: Tensor) -> tuple[NDArray[np.int8], NDArray[np.int32], float]:
    max_weight = float(np.max(np.abs(weights.data)))
    if max_weight == 0.0:
        raise ValueError("cannot quantize all-zero weights")
    scale = max_weight / 127.0
    quantized_weights = np.rint(weights.data / scale).clip(-127, 127).astype(np.int8)
    quantized_bias = np.rint(bias.data * (255.0 / scale)).astype(np.int32)
    return quantized_weights, quantized_bias, scale


def quantized_accuracy(
    images: Tensor,
    labels: NDArray[np.uint8],
    weights: NDArray[np.int8],
    bias: NDArray[np.int32],
) -> float:
    pixels = np.rint(images.matrix() * 255.0).astype(np.int32)
    scores = pixels @ weights.astype(np.int32).reshape(10, 784).T + bias
    return float((scores.argmax(axis=1) == labels).mean())


def serialize_model(weights: Tensor, bias: Tensor) -> bytes:
    if weights.shape != (10, 784) or bias.shape != (10,):
        raise ValueError("MNIST model must have weights [10, 784] and bias [10]")
    quantized_weights, quantized_bias, scale = quantize(weights, bias)
    layer_header = struct.pack("<HHBBH", 784, 10, 0, 0, 0)
    payload = (
        struct.pack("<f", scale)
        + quantized_weights.tobytes(order="C")
        + quantized_bias.astype("<i4", copy=False).tobytes(order="C")
    )
    total_size = HEADER_SIZE + LAYER_HEADER_SIZE + len(payload)
    header = struct.pack("<4sBBBBII", MAGIC, FORMAT_VERSION, DTYPE_INT8, 1, 0, total_size, 0)
    return header + layer_header + payload


def write_header(model: bytes, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as target:
        target.write(
            "#ifndef VAMPIREGRAD_GENERATED_MODEL_H\n"
            "#define VAMPIREGRAD_GENERATED_MODEL_H\n\n"
            "#include <stddef.h>\n"
            "#include <stdint.h>\n\n"
            "#ifdef __AVR__\n"
            "#include <avr/pgmspace.h>\n"
            "#define VAMPIREGRAD_MODEL_DATA PROGMEM\n"
            "#else\n"
            "#define VAMPIREGRAD_MODEL_DATA\n"
            "#endif\n\n"
            "static const uint8_t vampiregrad_model[] VAMPIREGRAD_MODEL_DATA = {\n    "
        )
        for index, byte in enumerate(model):
            target.write(f"0x{byte:02x}")
            if index + 1 != len(model):
                target.write(", ")
            if (index + 1) % 16 == 0 and index + 1 != len(model):
                target.write("\n    ")
        target.write(
            "\n};\n\n"
            f"#define vampiregrad_model_size ((size_t){len(model)}U)\n\n"
            "#undef VAMPIREGRAD_MODEL_DATA\n"
            "#endif\n"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".cache/mnist"))
    parser.add_argument("--output", type=Path, default=Path("model_data.h"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_mnist(args.cache)
    train_images = load_images(args.cache / "train-images-idx3-ubyte.gz", 60_000)
    train_labels = load_labels(args.cache / "train-labels-idx1-ubyte.gz", 60_000)
    test_images = load_images(args.cache / "t10k-images-idx3-ubyte.gz", 10_000)
    test_labels = load_labels(args.cache / "t10k-labels-idx1-ubyte.gz", 10_000)

    weights, bias = train(
        train_images,
        train_labels,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.seed,
    )
    quantized_weights, quantized_bias, scale = quantize(weights, bias)
    float_accuracy = accuracy(test_images, test_labels, weights, bias)
    int8_accuracy = quantized_accuracy(test_images, test_labels, quantized_weights, quantized_bias)
    print(f"test accuracy: float={float_accuracy:.2%}, int8={int8_accuracy:.2%}, scale={scale:.8f}")

    model = serialize_model(weights, bias)
    write_header(model, args.output)
    print(f"wrote {len(model)} bytes to {args.output}")


if __name__ == "__main__":
    main()
