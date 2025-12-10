import sys
from src.utils import read_data
from src.perceptron import initialize_weights_and_bias, train
from src.visualize import visualize_training

"""
Project: Perceptron Implementation
Author: Ntovonis Panagiotis
"""

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"[ERROR] Usage: python main.py <path_to_data_file>")
        sys.exit(1)
        
    filename = sys.argv[1]

    print(f"Loading data from {filename}...")
    data = read_data(filename)

    weights, bias = initialize_weights_and_bias(2)
    learning_rate = 0.01
    max_epochs = 100

    print("\nStarting training...")
    weights, bias, success = train(data, weights, bias, learning_rate, max_epochs)

    if success:
        print(f"\nTraining Finished")
        print("-Status: [SUCCESS]")
        print(f"-Final Weights: {weights}")
        print(f"-Final Bias: {bias}")
    else:
        print(f"Training Finished")
        print("Status: [FAILURE]")
        print(f"Failed to converge within {max_epochs} epochs.")
    
    visualize_training(filename, "data/history.txt")