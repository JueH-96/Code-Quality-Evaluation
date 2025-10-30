# YOUR CODE HERE
import sys

# Read input from stdin
input = sys.stdin.read
data = input().split()

# Parse input
N = int(data[0])
L = int(data[1])
R = int(data[2])

# Initialize sequence A
A = list(range(1, N + 1))

# Perform the operation of reversing the L-th through R-th elements
A[L-1:R] = reversed(A[L-1:R])

# Print the sequence after the operation
print(' '.join(map(str, A)))