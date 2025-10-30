# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
K = int(input[1])
A = list(map(int, input[2:]))

# Take K cards from the bottom and place them on top
result = A[-K:] + A[:-K]

# Print the result
print(" ".join(map(str, result)))