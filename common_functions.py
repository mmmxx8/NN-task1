from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from adaline import AdalineBinary
from perceptron import PerceptronBinary


def handle_missing_values(df):
    # Convert numeric-looking columns to floats if possible
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                pass

    # Use these grouping columns if available
    group_cols = [col for col in ["Species", "OriginLocation"] if col in df.columns]

    for col in df.columns:
        if df[col].isna().any() and col not in group_cols:
            if group_cols:
                df[col] = df.groupby(group_cols)[col].transform(
                    lambda x: x.fillna(x.mean())
                )
            # Fallback: fill any remaining missing values with global mean
            df[col] = df[col].fillna(df[col].mean())

    return df

def load_and_preprocess_data(feature_cols, class1, class2, add_bias=True):
    df = pd.read_csv("penguins.csv")
    df = handle_missing_values(df)

    # Filter only selected classes
    df = df[df["Species"].isin([class1, class2])].copy()

    # Encode classes
    le = LabelEncoder()
    df["Species"] = le.fit_transform(df["Species"])
    print(f"Class mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Balance samples (50 per class if possible)
    df = df.groupby("Species", group_keys=False).apply(lambda x: x.sample(min(50, len(x)), random_state=42))

    # Split train/test
    train_df = df.groupby("Species", group_keys=False).apply(lambda x: x.sample(30, random_state=42))
    test_df = df.drop(train_df.index)

    X_train = train_df[feature_cols].values
    y_train = train_df["Species"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["Species"].values

    # Normalize
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    # Add bias if needed
    if add_bias:
        X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

    return X_train, y_train, X_test, y_test, le

def run_nn_model(nn_type="perceptron"):
    # ----- User Inputs -----
    print("Available features: CulmenLength, CulmenDepth, FlipperLength, BodyMass")
    feature1 = input("Enter feature 1: ")
    feature2 = input("Enter feature 2: ")
    feature_cols = [feature1, feature2]

    print("\nAvailable classes: Adelie, Chinstrap, Gentoo")
    class1 = input("Enter first class (e.g., Adelie): ")
    class2 = input("Enter second class (e.g., Gentoo): ")

    eta = float(input("Enter learning rate (eta): "))
    m = int(input("Enter number of epochs (m): "))
    mse_threshold = float(input("Enter MSE threshold: "))
    add_bias = input("Add bias? (y/n): ").strip().lower() == "y"

    # ----- Data Preparation -----
    X_train, y_train, X_test, y_test, le = load_and_preprocess_data(feature_cols, class1, class2, add_bias)
    input_dim = X_train.shape[1]

    # ----- Model Selection -----
    if nn_type.lower() == "perceptron":
        model = PerceptronBinary(input_dim=input_dim, learning_rate=eta, n_epochs=m, mse_threshold=mse_threshold)
    elif nn_type.lower() == "adaline":
        model = AdalineBinary(input_dim=input_dim, learning_rate=eta, n_epochs=m, mse_threshold=mse_threshold)
    else:
        raise ValueError("Unknown NN type. Please choose 'perceptron' or 'adaline'.")

    # ----- Train Model -----
    model.train(X_train, y_train)

    # ----- Evaluation -----
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"\n🎯 Test Accuracy: {accuracy * 100:.2f}%")

    # ----- Visualization -----
    if hasattr(model, "errors_") and len(model.errors_) > 0:
        plt.figure(figsize=(6, 4))
        plt.plot(model.errors_, marker='o')
        plt.title(f"{nn_type.upper()} - MSE over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.grid(True)
        plt.show()

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_).plot(cmap='Blues')
    plt.title(f"{nn_type.upper()} Confusion Matrix")
    plt.show()

    # Decision Boundary (2D visualization)
    plt.figure(figsize=(8, 6))
    cmap_bg = ListedColormap(["#FFAAAA", "#AAFFAA"])
    cmap_pts = ListedColormap(["#FF0000", "#00AA00"])

    x_min, x_max = X_train[:, -2].min() - 1, X_train[:, -2].max() + 1
    y_min, y_max = X_train[:, -1].min() - 1, X_train[:, -1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    grid = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()] if add_bias else np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_bg)
    plt.scatter(X_test[:, -2], X_test[:, -1], c=y_test, cmap=cmap_pts, edgecolor='k', s=60)
    plt.xlabel(feature_cols[0])
    plt.ylabel(feature_cols[1])
    plt.title(f"{nn_type.upper()} Decision Boundary ({class1} vs {class2})")
    plt.show()

    return model, accuracy