"""Train a deep Keras neural network on the instructor-provided MNIST CSV files."""
from pathlib import Path
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Не знайдено {path}. Розпакуйте Lab03.zip у папку data/.")
    frame = pd.read_csv(path)
    if frame.shape[1] != 785 or frame.columns[0] != "label":
        raise ValueError(f"Неправильний формат {path.name}: очікується label + 784 пікселі")
    labels = frame.iloc[:, 0].to_numpy(dtype=np.int64)
    pixels = frame.iloc[:, 1:].to_numpy(dtype=np.float32) / 255.0
    if np.any((labels < 0) | (labels > 9)):
        raise ValueError("Мітки повинні бути цифрами від 0 до 9")
    return pixels, labels


def main():
    x_all, y_all = load_csv(DATA / "mnist_train.csv")
    x_test, y_test = load_csv(DATA / "mnist_test.csv")
    x_train, x_val, y_train, y_val = train_test_split(
        x_all, y_all, test_size=0.15, random_state=SEED, stratify=y_all
    )
    print(f"Навчальні: {len(x_train)} | Валідаційні: {len(x_val)} | Тестові: {len(x_test)}")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=15, batch_size=128, verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
    )
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    model.save(ROOT / "mnist_model.keras")
    print(f"Тестова точність: {accuracy:.4f}; втрати: {loss:.4f}")
    print("Переможець забирає все: argmax від 10 виходів softmax")
    epochs = np.arange(1, len(history.history["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history.history["loss"], marker="o", label="Навчальні втрати")
    ax.plot(epochs, history.history["val_loss"], marker="o", label="Валідаційні втрати")
    ax.set(xlabel="Епоха", ylabel="Loss", title="Залежність функції втрат від епохи")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "loss.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history.history["accuracy"], marker="o", label="Навчальна точність")
    ax.plot(epochs, history.history["val_accuracy"], marker="o", label="Валідаційна точність")
    ax.set(xlabel="Епоха", ylabel="Accuracy", title="Точність за епохами")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "accuracy.png", dpi=170)
    plt.close(fig)
    sample_ids = np.random.default_rng(SEED).choice(len(x_test), size=12, replace=False)
    probs = model.predict(x_test[sample_ids], verbose=0)
    predictions = np.argmax(probs, axis=1)
    fig, axes = plt.subplots(3, 4, figsize=(9, 7))
    for ax, image, pred, true in zip(axes.flat, x_test[sample_ids], predictions, y_test[sample_ids]):
        ax.imshow(image.reshape(28, 28), cmap="gray")
        ax.set_title(f"Прогноз: {pred} | Факт: {true}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "predictions.png", dpi=170)
    plt.close(fig)
    (RESULTS / "metrics.json").write_text(json.dumps({
        "train_count": len(x_train), "validation_count": len(x_val),
        "test_count": len(x_test), "epochs_completed": len(epochs),
        "test_accuracy": float(accuracy), "test_loss": float(loss),
        "winner_takes_all": "argmax(softmax_outputs)",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Модель і графіки збережено. Запустіть: python -m streamlit run app.py")

if __name__ == "__main__":
    main()
