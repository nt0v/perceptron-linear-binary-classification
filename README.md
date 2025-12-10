Title: "**Perceptron: Linear Binary Classification**"

Single-Layer Perceptron Implementation for Linear Binary Classification.

**Problem Description:**<br>
This program solves the problem of separating two classes of data points 
by finding a linear decision boundary (hyperplane).  

Based on the geometric interpretation of the Perceptron:<br>  
- The algorithm iteratively adjusts the weight vector 'w' and bias 'b'.
- These parameters define a separating hyperplane H(w).
- The goal is to orient H(w) such that:
    * All points of Class 1 lie in the positive half-space (u(x) > 0).
    * All points of Class -1 lie in the negative half-space (u(x) < 0).
 
**Limitations:**<br>
The algorithm is guaranteed to converge and solve the problem ONLY if 
the dataset is linearly separable. For non-separable data (e.g., XOR), 
it will fail to find a perfect solution, and will stop after it reaches
a max value of epochs.
