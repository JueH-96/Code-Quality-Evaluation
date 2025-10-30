# Read the input from stdin
import sys

# Read the first line
N, K, X = map(int, sys.stdin.readline().split())

# Read the second line
A = list(map(int, sys.stdin.readline().split()))

# Insert X after the K-th element of A
B = A[:K] + [X] + A[K:]

# Print the result
print(*B)