import numpy as np

# Training data
X = np.array([[1, 1, 1, -1],
              [-1, -1, 1, 1]])
Y = np.array([[1, -1],
              [-1, 1]])

# Weight matrix using outer product sum
W = np.dot(Y.T, X)

# Sign function that handles zero
def custom_sign(x):
    return np.where(x >= 0, 1, -1)

# BAM recall function (iterates until convergence)
def bam_bidirectional(x_init, max_iters=10):
    x = x_init.copy()
    for _ in range(max_iters):
        y = custom_sign(np.dot(W, x))          # X to Y
        x_new = custom_sign(np.dot(W.T, y))    # Y to X
        if np.array_equal(x, x_new):           # Check for convergence
            break
        x = x_new
    return x, y

# Test input
x_test = np.array([1, -1, -1, -1])

# Run BAM
x_final, y_final = bam_bidirectional(x_test)

# Print results
print("Initial input x:", x_test)
print("Final recalled x:", x_final)
print("Recalled y:", y_final)
