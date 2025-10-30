# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:]))

# Take K cards from the bottom and place them on top
B = A[-K:] + A[:-K]

# Print the result
print(" ".join(map(str, B)))