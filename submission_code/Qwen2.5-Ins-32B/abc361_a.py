import sys

# Read input from stdin
input = sys.stdin.read
data = input().split()

# Convert input strings to integers
N, K, X = map(int, data[0:3])
A = list(map(int, data[3:]))

# Insert X immediately after the K-th element
B = A[:K] + [X] + A[K:]

# Print the resulting sequence
print(' '.join(map(str, B)))