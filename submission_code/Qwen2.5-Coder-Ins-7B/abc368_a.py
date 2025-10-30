# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:]))

# Take the last K elements and move them to the front
result = A[-K:] + A[:-K]

# Print the result
print(' '.join(map(str, result)))