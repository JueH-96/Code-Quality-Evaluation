import math
import sys

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A = [0] * (N + 1)
for i in range(1, N + 1):
    A[i] = int(data[index])
    index += 1

# Define V and W
V = [0] * (N + 1)
W = [0] * (N + 1)
for i in range(1, N + 1):
    V[i] = A[i] + i
    W[i] = A[i] - i

# Compute log_val for RMQ
log_val = [0] * (N + 1)
for i in range(2, N + 1):
    log_val[i] = log_val[i // 2] + 1

# Maximum log value for sparse table
max_log = int(math.log2(N)) + 1

# Build sparse table for V
st_V = [[0] * (max_log + 1) for _ in range(N + 1)]
for i in range(1, N + 1):
    st_V[i][0] = V[i]
for j in range(1, max_log + 1):
    max_i = N - (1 << j) + 1
    if max_i >= 1:
        for i in range(1, max_i + 1):
            st_V[i][j] = min(st_V[i][j - 1], st_V[i + (1 << (j - 1))][j - 1])

# Build sparse table for W
st_W = [[0] * (max_log + 1) for _ in range(N + 1)]
for i in range(1, N + 1):
    st_W[i][0] = W[i]
for j in range(1, max_log + 1):
    max_i = N - (1 << j) + 1
    if max_i >= 1:
        for i in range(1, max_i + 1):
            st_W[i][j] = min(st_W[i][j - 1], st_W[i + (1 << (j - 1))][j - 1])

# Define range minimum query function
def rmq_min(st, L, R):
    len_range = R - L + 1
    k = log_val[len_range]
    return min(st[L][k], st[R - (1 << k) + 1][k])

# Initialize answer
ans = 0

# Iterate over each possible center
for c in range(1, N + 1):
    # Binary search for maximum K_right
    left = 1
    right = N
    res_right = 0
    while left <= right:
        mid = (left + right) // 2
        if c + mid - 1 <= N and rmq_min(st_V, c, c + mid - 1) >= mid + c:
            res_right = mid
            left = mid + 1
        else:
            right = mid - 1
    
    # Binary search for maximum K_left
    left = 1
    right = N
    res_left = 0
    while left <= right:
        mid = (left + right) // 2
        if c - mid + 1 >= 1 and rmq_min(st_W, c - mid + 1, c) >= mid - c:
            res_left = mid
            left = mid + 1
        else:
            right = mid - 1
    
    # Compute k for this center
    k_c = min(res_right, res_left)
    if k_c > ans:
        ans = k_c

# Output the answer
print(ans)