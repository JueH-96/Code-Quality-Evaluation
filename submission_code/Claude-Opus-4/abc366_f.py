# YOUR CODE HERE
from itertools import permutations

N, K = map(int, input().split())
functions = []
for _ in range(N):
    A, B = map(int, input().split())
    functions.append((A, B))

max_value = 0

# Generate all permutations of K indices from N indices
for perm in permutations(range(N), K):
    # Start with x = 1
    x = 1
    # Apply functions in the order specified by the permutation
    for idx in perm:
        A, B = functions[idx]
        x = A * x + B
    max_value = max(max_value, x)

print(max_value)