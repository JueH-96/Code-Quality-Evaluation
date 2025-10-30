# YOUR CODE HERE
import sys

# Read input from stdin
input_data = sys.stdin.read().strip()

# Convert the input string to a list of integers
A = list(map(int, input_data.split()))

# Calculate the result using the formula
result = sum(A[i] * (2 ** i) for i in range(64))

# Print the result
print(result)