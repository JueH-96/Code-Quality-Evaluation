from itertools import permutations

n, k = map(int, input().split())
functions = []
for _ in range(n):
    a, b = map(int, input().split())
    functions.append((a, b))

max_value = float('-inf')

# Generate all permutations of k distinct integers from 0 to n-1 (0-indexed)
for perm in permutations(range(n), k):
    # Start with x = 1
    value = 1
    # Apply functions in reverse order (from p_k to p_1)
    for i in range(k-1, -1, -1):
        a, b = functions[perm[i]]
        value = a * value + b
    max_value = max(max_value, value)

print(max_value)