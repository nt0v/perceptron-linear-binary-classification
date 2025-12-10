# Perceptron: Linear Binary Classification

Single-Layer Perceptron Implementation for Linear Binary Classification in Python. Built from scratch to demonstrate the mathematical foundations of neural networks without relying on high-level ML libraries.

<p align="center">
  <img src="img/perceptron_training.gif" alt="Perceptron Training Animation" width="600">
</p>

---

## Problem Description

This program solves the problem of separating two classes of data points by finding a linear decision boundary (hyperplane).

Based on the geometric interpretation of the Perceptron:
* The algorithm iteratively adjusts the weight vector **'w'** and bias **'b'**.
* These parameters define a separating hyperplane **H(w)**.
* The goal is to orient H(w) such that:
    * All points of Class 1 lie in the positive half-space (`u(x) > 0`).
    * All points of Class -1 lie in the negative half-space (`u(x) < 0`).

### Limitations
The algorithm is guaranteed to converge and solve the problem **ONLY if the dataset is linearly separable**. For non-separable data (e.g., XOR), it will fail to find a perfect solution, and will stop after it reaches a max value of epochs.<br>

<p align="center">
  <img src="img/perceptron.png" alt="Perceptron" width="750">
</p>

---

## Getting Started

To run this project on your local machine, follow these steps:

### 1. Prerequisites
Make sure you have **Python 3.x** installed. You will also need `numpy` and `matplotlib` for the visualization.

Install the dependencies using pip:

```bash
pip install numpy matplotlib
```

### 2. Running the Program
The main script takes the path to a dataset file as an argument.

Run the following command in your terminal:

```bash
python main.py data/dataset_hard.txt
```

*(Replace `data/dataset_hard.txt` with your own file path if needed)*

---

## Input Data Format

If you want to train the Perceptron on your own data, create a `.txt` file where each line represents a data point. The format must be **space-separated** with exactly 3 columns:

```text
x1 x2 label
```
* **x1**: The first feature (float).
* **x2**: The second feature (float).
* **label**: The class label. Must be strictly **1** or **-1**.

**Example:**
```text
1.5 2.2 1
-0.5 3.1 -1
2.0 1.0 1
```

---

## Author

**Ntovonis Panagiotis**
* Email: panag.ntov@gmail.com
* GitHub: [nt0v](https://github.com/nt0v)

---

## License

This project is open-source and available for educational purposes.
