from common_functions import run_nn_model


if __name__ == "__main__":
    print("=== Neural Network Trainer ===")
    print("Available NN types: perceptron | adaline")
    
    nn_type = input("Enter NN type: ").strip().lower()
    if nn_type not in ["perceptron", "adaline"]:
        print("❌ Invalid NN type. Please enter 'perceptron' or 'adaline'.")
    else:
        model, acc = run_nn_model(nn_type)
        print(f"\n✅ {nn_type.upper()} model completed with accuracy: {acc * 100:.2f}%")
