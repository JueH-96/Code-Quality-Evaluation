import sys

# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Initialize count of possible pairs
count = 0

# Iterate through all possible pairs of rest areas
for s in range(1, N+1):
    for t in range(1, N+1):
        if s != t:
            # Calculate the minimum number of steps to walk clockwise from s to t
            steps = 0
            for i in range(s, t):
                steps += A[i-1]
            if steps % M == 0:
                count += 1

# Print the answer
print(count)