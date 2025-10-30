import sys

# Define Fenwick Tree functions
def add(ft, idx, val=1):
    idx_val = idx
    while idx_val <= MAX_VAL:
        ft[idx_val] += val
        idx_val += idx_val & -idx_val

def prefix_sum(ft, idx):
    sum_res = 0
    i = idx
    while i > 0:
        sum_res += ft[i]
        i -= i & -i
    return sum_res

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A = list(map(int, data[index:index + N]))

# Set maximum value for Fenwick Tree index
MAX_VAL = 2000005
ft = [0] * (MAX_VAL + 1)

# List to store the final stone counts
ans = [0] * N

# Compute for each alien
for k in range(1, N + 1):  # k is 1-based
    if k == 1:
        R = 0
    else:
        total_prev = k - 1
        num_less_k = prefix_sum(ft, k - 1)  # Number with val_j <= k-1
        R = total_prev - num_less_k
    S = A[k - 1] + R  # S_k
    # Compute final stones B_k
    B_k = max(0, S - (N - k))
    ans[k - 1] = B_k
    # Compute val_k and add to Fenwick tree
    val_k = S + k
    add(ft, val_k)

# Output the result
print(*ans)