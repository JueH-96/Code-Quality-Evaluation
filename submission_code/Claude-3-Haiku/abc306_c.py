from collections import defaultdict

# Read input
N = int(input())
A = list(map(int, input().split()))

# Compute the middle index for each number
f = {}
for i in range(1, N+1):
    indices = []
    for j in range(len(A)):
        if A[j] == i:
            indices.append(j)
    f[i] = indices[1]

# Sort the numbers by their middle index
result = sorted(range(1, N+1), key=lambda x: f[x])

# Print the result
print(" ".join(map(str, result)))