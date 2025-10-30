# YOUR CODE HERE
import sys
sys.setrecursionlimit(1000000)  # This might not be necessary, but can be added if needed for large N, though N<=21 so not required.

# Read N from standard input
N = int(input())

# Iterate through all possible x, y, z such that x + y + z <= N in lex order
for x in range(0, N + 1):
    for y in range(0, N - x + 1):
        for z in range(0, N - x - y + 1):
            print(x, y, z)