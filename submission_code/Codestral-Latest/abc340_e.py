# YOUR CODE HERE
import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

A = list(map(int, data[2:2+N]))
B = list(map(int, data[2+N:2+N+M]))

# Initialize the result array with the initial number of balls
result = A.copy()

# Process each operation
for i in range(M):
    bi = B[i]
    num_balls = result[bi]
    result[bi] = 0

    # Distribute the balls
    for j in range(num_balls):
        result[(bi + j) % N] += 1

# Print the result
print(" ".join(map(str, result)))