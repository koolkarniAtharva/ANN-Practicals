import numpy as np

# Step function (activation)
step_function = lambda x: 1 if x >= 0 else 0

# ASCII training data for digits '0' to '9'
# ASCII of '0' = 48 (binary: 0110000), up to '9' = 57 (binary: 0111001)
training_data = []

for i in range(10):
    char = chr(48 + i)  # Get ASCII character for digits 0 to 9
    ascii_value = ord(char)
    binary_input = [int(bit) for bit in '{0:07b}'.format(ascii_value)]  # 7-bit binary
    label = 1 if i % 2 == 0 else 0  # 1 = even, 0 = odd
    training_data.append({'input': binary_input, 'label': label})

# Initialize weights (same size as input vector)
weights = np.zeros(7)

# Training loop (single pass)
for data in training_data:
    input_vector = np.array(data['input'])
    label = data['label']
    output = step_function(np.dot(input_vector, weights))
    error = label - output
    weights += input_vector * error  # Perceptron update rule

# User input
user_char = input("Enter a digit between 0 and 9: ")

# Validate input
if user_char not in "0123456789":
    print("Invalid input. Please enter a digit between 0 and 9.")
else:
    ascii_input = ord(user_char)
    binary_input = np.array([int(bit) for bit in '{0:07b}'.format(ascii_input)])
    prediction = step_function(np.dot(binary_input, weights))
    result = "even" if prediction == 1 else "odd"
    print(f"{user_char} (ASCII: {ascii_input}) is {result}.")
