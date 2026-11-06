"""Streamlit interface for the trained handwritten-digit classifier."""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="MNIST | Нейронна мережа", page_icon="🔢", layout="wide")
st.title("🔢 Розпізнавання рукописних цифр")
st.caption("Глибока нейронна мережа Keras · MNIST 28×28 · принцип «переможець забирає все»")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(ROOT / "mnist_model.keras")

if not (ROOT / "mnist_model.keras").exists():
    st.warning("Спочатку навчіть модель командою: python train.py")
    st.stop()
model = load_model()

def predict(pixels):
    probabilities = model.predict(pixels.reshape(1, 784), verbose=0)[0]
    return int(np.argmax(probabilities)), probabilities

st.sidebar.header("Джерело зображення")
mode = st.sidebar.radio("Оберіть режим", ["Тестовий приклад із CSV", "Завантажити власне зображення"])
if mode == "Тестовий приклад із CSV":
    csv_path = ROOT / "data" / "mnist_test.csv"
    if not csv_path.exists():
        st.error("Не знайдено data/mnist_test.csv")
        st.stop()
    @st.cache_data
    def read_examples():
        return pd.read_csv(csv_path, nrows=500)
    df = read_examples()
    idx = st.slider("Номер тестового прикладу", 0, len(df) - 1, 0)
    label = int(df.iloc[idx, 0])
    pixels = df.iloc[idx, 1:].to_numpy(dtype=np.float32) / 255.0
    st.info(f"Справжня цифра: {label}")
else:
    st.info("Завантажте одну темну цифру на світлому тлі або світлу цифру на темному тлі. Для кращого результату використовуйте зображення, подібне до MNIST.")
    uploaded = st.file_uploader("Зображення PNG/JPG", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.stop()
    gray = Image.open(uploaded).convert("L")
    invert = st.checkbox("Інвертувати кольори (темна цифра на білому тлі)", value=True)
    if invert:
        gray = ImageOps.invert(gray)
    gray = ImageOps.contain(gray, (20, 20))
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(gray, ((28 - gray.width) // 2, (28 - gray.height) // 2))
    pixels = np.asarray(canvas, dtype=np.float32).reshape(784) / 255.0
    st.caption("Попередня обробка: відтінки сірого, інверсія за потреби, масштабування та центрування до 28×28.")

prediction, probabilities = predict(pixels)
left, right = st.columns([1, 2])
with left:
    st.image(pixels.reshape(28, 28), width=240, clamp=True, caption="Вхід 28×28")
    st.metric("Розпізнана цифра", prediction)
    st.metric("Максимальний вихід softmax", f"{probabilities[prediction]:.1%}")
with right:
    st.subheader("Виходи нейронної мережі")
    st.bar_chart(pd.DataFrame({"Ймовірність": probabilities}, index=[str(i) for i in range(10)]))
    st.code(f"prediction = np.argmax(probabilities)  # {prediction}", language="python")
    st.caption("Клас обирається за найбільшим виходом softmax (winner-takes-all).")

st.divider()
st.subheader("Графіки навчання")
for filename, title in [("loss.png", "Функція втрат"), ("accuracy.png", "Точність"), ("predictions.png", "Приклади прогнозів")]:
    path = ROOT / "results" / filename
    if path.exists():
        st.image(str(path), caption=title, use_container_width=True)
