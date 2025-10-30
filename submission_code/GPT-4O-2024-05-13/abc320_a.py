# YOUR CODE HERE
import sys

# Read input from standard input
input_line = sys.stdin.read().strip()
A, B = map(int, input_line.split())

# Calculate A^B and B^A
result = A**B + B**A

# Print the result
print(result)