import sys

# Read input from stdin
N, Q = map(int, sys.stdin.readline().split())
queries = list(map(int, sys.stdin.readline().split()))

# Initialize the sequence A and the set S
A = [0] * N
S = set()

# Process each query
for x in queries:
    if x in S:
        S.remove(x)
    else:
        S.add(x)
    for j in S:
        A[j - 1] += len(S)

# Print the sequence A to stdout
print(' '.join(map(str, A)))