import math
import sys

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A_values = [int(data[index + i]) for i in range(N)]
index += N

# Make A 1-based
A = [0]  # Index 0 unused
A.extend(A_values)  # A[1] to A[N]

# Compute prefix sum P
P = [0] * (N + 1)
for i in range(1, N + 1):
    P[i] = P[i - 1] + A[i]

# Compute B[i] = P[i] + A[i] for i=1 to N
B = [0] * (N + 1)
for i in range(1, N + 1):
    B[i] = P[i] + A[i]

# Compute C[j] = P[j-1] - A[j] for j=2 to N
C = [0] * (N + 1)  # Unused at index 0 and 1
for j in range(2, N + 1):
    C[j] = P[j - 1] - A[j]

# Compute LOGN
LOGN = int(math.log2(N)) + 1

# Build sparse table for max of B
sparse_max_B = [[0 for _ in range(N + 1)] for _ in range(LOGN)]
for i in range(1, N + 1):
    sparse_max_B[0][i] = B[i]
for k in range(1, LOGN):
    max_i_index = N - (1 << k) + 1
    if max_i_index >= 1:
        for i in range(1, max_i_index + 1):
            sparse_max_B[k][i] = max(sparse_max_B[k - 1][i], sparse_max_B[k - 1][i + (1 << (k - 1))])

# Build sparse table for min of C, indices from 2 to N
sparse_min_C = [[0 for _ in range(N + 1)] for _ in range(LOGN)]
for i in range(2, N + 1):
    sparse_min_C[0][i] = C[i]
for k in range(1, LOGN):
    max_i_index_C = N - (1 << k) + 1
    min_i_C = 2
    if max_i_index_C >= min_i_C:
        for i in range(min_i_C, max_i_index_C + 1):
            sparse_min_C[k][i] = min(sparse_min_C[k - 1][i], sparse_min_C[k - 1][i + (1 << (k - 1))])

# Function to query range max for B
def range_max_B_func(left, right):
    if left > right:
        return float('-inf')
    range_len = right - left + 1
    k_log = int(math.log2(range_len))
    return max(sparse_max_B[k_log][left], sparse_max_B[k_log][right - (1 << k_log) + 1])

# Function to query range min for C
def range_min_C_func(left, right):
    if left > right:
        return float('inf')
    range_len = right - left + 1
    k_log = int(math.log2(range_len))
    return min(sparse_min_C[k_log][left], sparse_min_C[k_log][right - (1 << k_log) + 1])

# List to store answers
ans_list = []

# For each K from 1 to N
for K in range(1, N + 1):
    # Binary search for left L_start
    low = 1
    high = K
    while low <= high:
        mid = (low + high) // 2
        if (mid > K - 1) or (range_max_B_func(mid, K - 1) < P[K]):
            high = mid - 1
        else:
            low = mid + 1
    L_start = low  # First m where condition true

    # Binary search for right R_end
    low_r = K
    high_r = N
    ans_r = K
    while low_r <= high_r:
        mid_r = (low_r + high_r) // 2
        if (mid_r < K + 1) or (range_min_C_func(K + 1, mid_r) > P[K - 1]):
            ans_r = mid_r
            low_r = mid_r + 1  # Try larger
        else:
            high_r = mid_r - 1  # Try smaller
    R_end = ans_r

    # Compute sum from L_start to R_end
    sum_size = P[R_end] - P[L_start - 1]
    ans_list.append(sum_size)

# Output the answers
print(' '.join(map(str, ans_list)))