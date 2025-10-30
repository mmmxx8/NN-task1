import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from common_functions import load_and_preprocess_data
from perceptron import PerceptronBinary
from adaline import AdalineBinary

st.set_page_config(page_title="Neural Network Trainer", layout="centered")

st.title(" Neural Network Trainer (Perceptron & Adaline)")

# --- User Inputs ---
nn_type = st.radio("Select Algorithm:", ["Perceptron", "Adaline"])
feature1 = st.selectbox("Feature 1:", ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"])
feature2 = st.selectbox("Feature 2:", ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"])
class1 = st.selectbox("Class 1:", ["Adelie", "Chinstrap", "Gentoo"])
class2 = st.selectbox("Class 2:", ["Adelie", "Chinstrap", "Gentoo"])

eta = st.number_input("Learning Rate (η):", value=0.01, min_value=0.0001, step=0.001)
epochs = st.number_input("Epochs:", value=100, min_value=10, step=10)
mse_threshold = st.number_input("MSE Threshold:", value=0.001, min_value=0.0001, step=0.001)
add_bias = st.checkbox("Add Bias", value=True)

if st.button(" Start Training"):
    # --- Data ---
    X_train, y_train, X_test, y_test, le = load_and_preprocess_data([feature1, feature2], class1, class2, add_bias)
    input_dim = X_train.shape[1]

    # --- Model ---
    if nn_type.lower() == "perceptron":
        model = PerceptronBinary(input_dim, eta, epochs, mse_threshold, add_bias)
    else:
        model = AdalineBinary(input_dim, eta, epochs, mse_threshold)

    # --- Train ---
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- confusion metrix ---
    cm = np.zeros((2, 2), dtype=int)
    for yt, yp in zip(y_test, y_pred):
        cm[int(yt)][int(yp)] += 1
    accuracy = np.trace(cm) / np.sum(cm)

    st.success(f" Accuracy: {accuracy*100:.2f}%")

    # --- Confusion Matrix Display ---
    st.subheader("Confusion Matrix")
    st.write("Rows = Actual Classes | Columns = Predicted Classes")

    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')

    # labels
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(le.classes_)
    ax.set_yticklabels(le.classes_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    # text values inside cells
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    st.pyplot(fig)

    # --- MSE Plot ---
    if hasattr(model, "errors_") and len(model.errors_) > 0:
        fig, ax = plt.subplots()
        ax.plot(model.errors_, marker='o')
        ax.set_title(f"{nn_type} - MSE over Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        st.pyplot(fig)

    # --- Decision Boundary ---
    st.subheader("Decision Boundary Visualization")
    fig, ax = plt.subplots()
    x_min, x_max = X_train[:, -2].min() - 1, X_train[:, -2].max() + 1
    y_min, y_max = X_train[:, -1].min() - 1, X_train[:, -1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()] if add_bias else np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X_test[:, -2], X_test[:, -1], c=y_test, cmap='bwr', edgecolor='k')
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.set_title(f"{nn_type} Decision Boundary")
    st.pyplot(fig)
