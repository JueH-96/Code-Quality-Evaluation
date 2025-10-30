import itertools

N, K = map(int, input().split())
functions = []
for _ in range(N):
    A, B = map(int, input().split())
    functions.append((A, B))

max_value = 0

# Try all permutations of K functions from N functions
for perm in itertools.permutations(range(N), K):
    value = 1
    # Apply functions from right to left in the composition
    for i in reversed(perm):
        A, B = functions[i]
        value = A * value + B
    max_value = max(max_value, value)

print(max_value)