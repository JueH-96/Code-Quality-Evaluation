# YOUR CODE HERE
import sys

# Read input from stdin
input = sys.stdin.read
data = input().split()

# Convert input data to integers
A = [int(x) for x in data]

# Print the elements in reverse order
for i in range(len(A) - 1, -1, -1):
    print(A[i])