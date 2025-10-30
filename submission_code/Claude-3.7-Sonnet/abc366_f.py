from itertools import permutations

N, K = map(int, input().split())
functions = []
for _ in range(N):
    A, B = map(int, input().split())
    functions.append((A, B))

def compute_value(functions, sequence):
    value = 1
    for i in reversed(sequence):
        A, B = functions[i - 1]  # Adjusting for 0-based indexing
        value = A * value + B
    return value

max_value = float('-inf')
for p in permutations(range(1, N + 1), K):
    value = compute_value(functions, p)
    max_value = max(max_value, value)

print(max_value)