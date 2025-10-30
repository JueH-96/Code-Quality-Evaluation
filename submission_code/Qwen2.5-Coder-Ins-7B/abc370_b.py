# YOUR CODE HERE
import sys

# Read input from stdin
input = sys.stdin.read
data = input().split()

# Parse the input
N = int(data[0])
A = []
index = 1
for i in range(N):
    row = list(map(int, data[index:index + N]))
    A.append(row)
    index += N

# Initialize the current element to 1
current = 1

# Combine the current element with each subsequent element
for i in range(1, N):
    current = A[current - 1][i - 1]

# Print the final element
print(current)