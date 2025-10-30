# YOUR CODE HERE
import sys

# Read the input from standard input
input_data = sys.stdin.read().strip()

# Split the input into a list of integers
A = list(map(int, input_data.split()))

# Calculate the result using bitwise operations
result = 0
for i in range(64):
    result += A[i] * (1 << i)

# Print the result
print(result)