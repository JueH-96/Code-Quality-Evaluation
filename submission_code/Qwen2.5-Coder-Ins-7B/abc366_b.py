# YOUR CODE HERE
import sys

# Read input
input = sys.stdin.read
data = input().split()

# Number of strings
N = int(data[0])

# List of strings
S = data[1:]

# Find the maximum length of the strings
M = max(len(s) for s in S)

# Initialize the result list
result = [''] * M

# Fill the result list
for i in range(M):
    for j in range(N):
        if i < len(S[j]):
            result[i] += S[j][i]
        else:
            result[i] += '*'

# Print the result
for s in result:
    print(s)