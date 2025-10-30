# YOUR CODE HERE
import sys

# Read input
N = int(input())
S = list(input())
Q = int(input())

# Process each operation
for _ in range(Q):
    c, d = input().split()
    for i in range(N):
        if S[i] == c:
            S[i] = d

# Print the final string
print(''.join(S))