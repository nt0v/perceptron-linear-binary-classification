"""
About Me:
-Fullname: Ntovonis Panagiotis
-email: panag.ntov@gmail.com

Single-Layer Perceptron Implementation for Linear Binary Classification.

Problem Description:
    This program solves the problem of separating two classes of data points 
    by finding a linear decision boundary (hyperplane).
    
    Based on the geometric interpretation of the Perceptron:
    - The algorithm iteratively adjusts the weight vector 'w' and bias 'b'.
    - These parameters define a separating hyperplane H(w).
    - The goal is to orient H(w) such that:
        * All points of Class 1 lie in the positive half-space (u(x) > 0).
        * All points of Class -1 lie in the negative half-space (u(x) < 0).

Limitations:
    The algorithm is guaranteed to converge and solve the problem ONLY if 
    the dataset is linearly separable. For non-separable data (e.g., XOR), 
    it will fail to find a perfect solution, and will stop after it reaches
    a max value of epochs.
"""

import random
import sys
import time


def initialize_weights_and_bias(d):
    """
    Generates random weights and bias uniformly distributed between -1 and 1.
    Returns:
        (list[float], float): A tuple containing the initialized weights list 
        of size 'd' and the bias scalar.
    """
    weights = []
    for _ in range(d):
        weights.append((2 * random.random()) -1)
    bias = (2 * random.random()) - 1
    return weights, bias


def calculate_neuron_input(data, weights, bias):
    """
    Calculates the linear weighted sum of inputs including the bias.
    Returns:
        float: The computed net input value (u).
    """
    u = 0
    for i in range(len(weights)):
        u += data[i] * weights[i]
    u += bias
    return u


def activation_function(u):
    """
    Applies a threshold function to classification scores.
    Returns:
        int: 1 for positive input, -1 for zero or negative input.
    """
    if (u <= 0):
        return -1
    return 1


def train(data, weights, bias, learning_rate, max_epochs):
    """
    Executes the perceptron training loop until weights stabilize (convergence).
    Updates model parameters based on misclassifications and prints the total 
    number of epochs required upon completion.
    Returns:
        (list, float): The optimized weights and bias after training.
    """
    history_file = open("history.txt", "w")    
    k = 0
    total_updates = 0
    start = time.time()
    while (k < max_epochs):        
        error_count = 0
        for i in range(len(data)):
            point = data[i][0]
            t = data[i][1]
            u = calculate_neuron_input(point, weights, bias)
            o = activation_function(u)
            error = t - o
            if (error != 0):
                error_count += 1
                total_updates += 1
                for j in range(len(weights)):
                    weights[j] = weights[j] + (learning_rate * error * point[j])
                bias = bias + (learning_rate * error)
            history_file.write(f"{k} {i} {weights[0]} {weights[1]} {bias} {total_updates}\n")
        if error_count == 0:
            end = time.time()
            duration = end - start
            print(f"Training Finished")
            print(f"-Total Epochs: {k}")
            print(f"-Time: {duration} seconds")
            history_file.close()
            return weights, bias, True          
        k += 1
    history_file.close()
    return weights, bias, False