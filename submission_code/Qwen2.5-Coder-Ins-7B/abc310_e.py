# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]

# Convert S to a list of integers
A = [int(char) for char in S]

# Initialize the result
result = 0

# Iterate over all possible i and j
for i in range(N):
    current = A[i]
    for j in range(i, N):
        current = (current & ~A[j]) | A[j]
        result += current

# Print the result
print(result)