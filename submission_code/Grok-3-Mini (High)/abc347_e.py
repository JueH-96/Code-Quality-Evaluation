import sys

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
Q = int(data[index])
index += 1
x = [int(data[i]) for i in range(index, index + Q)]

# Simulate to get sz_after
in_set = [False] * (N + 1)
size_s = 0
sz_after = [0] * Q
for idx in range(Q):
    val = x[idx]
    if in_set[val]:
        in_set[val] = False
        size_s -= 1
    else:
        in_set[val] = True
        size_s += 1
    sz_after[idx] = size_s

# Compute prefix sum of sz_after
prefix_sum_sz = [0] * (Q + 1)
for i in range(Q):
    prefix_sum_sz[i + 1] = prefix_sum_sz[i] + sz_after[i]

# Build positions where each j is toggled
pos = [[] for _ in range(N + 1)]
for idx in range(Q):
    val = x[idx]
    pos[val].append(idx)

# Compute A for each j
A = [0] * (N + 1)
for j in range(1, N + 1):
    p = pos[j]
    k = len(p)
    for i in range(0, k, 2):  # Iterate over even indices
        left = p[i]
        if i + 1 < k:
            right = p[i + 1] - 1
        else:
            right = Q - 1
        sum_val = prefix_sum_sz[right + 1] - prefix_sum_sz[left]
        A[j] += sum_val

# Output the result
print(' '.join(map(str, A[1:])))