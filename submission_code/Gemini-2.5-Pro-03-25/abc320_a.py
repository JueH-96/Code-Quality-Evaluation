# YOUR CODE HERE
import sys

# Read input from stdin and map to integers
# The input consists of two integers A and B separated by a space on a single line.
A, B = map(int, sys.stdin.readline().split())

# Calculate A raised to the power of B (A^B)
a_pow_b = A ** B

# Calculate B raised to the power of A (B^A)
b_pow_a = B ** A

# Calculate the sum A^B + B^A
result = a_pow_b + b_pow_a

# Print the final result to stdout
print(result)