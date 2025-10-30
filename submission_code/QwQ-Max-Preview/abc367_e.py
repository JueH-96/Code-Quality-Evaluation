n, k = map(int, input().split())
x = list(map(int, input().split()))
a = list(map(int, input().split()))

MAX_LOG = 60

# Convert X to 0-based indices and create the f array
f = [xi - 1 for xi in x]

# Initialize binary lifting table
up = [[0] * n for _ in range(MAX_LOG)]

# Base case: up[0][i] is the direct parent (f[i])
for i in range(n):
    up[0][i] = f[i]

# Fill the binary lifting table
for j in range(1, MAX_LOG):
    for i in range(n):
        up[j][i] = up[j-1][up[j-1][i]]

# Compute the K-th ancestor for each position
ancestor = list(range(n))
remaining = k
for j in reversed(range(MAX_LOG)):
    if remaining >= (1 << j):
        remaining -= (1 << j)
        for i in range(n):
            ancestor[i] = up[j][ancestor[i]]

# Generate the result array
result = [a[ancestor[i]] for i in range(n)]
print(' '.join(map(str, result)))