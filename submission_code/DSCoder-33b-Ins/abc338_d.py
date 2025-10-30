# YOUR CODE HERE

import sys

def solve(N, M, X):
    # Create a list of bridges
    bridges = [0] * N
    for i in range(M):
        bridges[X[i]-1] += 1

    # Calculate the minimum length of the tour
    min_length = N
    for i in range(N):
        length = sum(bridges[i:]) + sum(bridges[:i])
        min_length = min(min_length, length)

    return min_length

# Read the inputs
N, M = map(int, sys.stdin.readline().split())
X = list(map(int, sys.stdin.readline().split()))

# Solve the problem
print(solve(N, M, X))