import sys

def popcount(x, W):
    return sum((x >> i) & 1 for i in range(W))

def min_hw_diff(d, W):
    pc = popcount(d, W)
    return min(pc, W - pc)

def fwht(arr):
    arr_cp = list(arr)  # Make a copy
    N = len(arr_cp)
    W_log = N.bit_length() - 1  # W
    for i in range(W_log):
        block_size = 1 << i
        for start in range(0, N, 1 << (i + 1)):
            for idx in range(block_size):
                u_idx = start + idx
                v_idx = start + idx + block_size
                u = arr_cp[u_idx]
                v = arr_cp[v_idx]
                arr_cp[u_idx] = u + v
                arr_cp[v_idx] = u - v
    return arr_cp

# Read input
data = sys.stdin.read().split()
index = 0
H = int(data[index])
W = int(data[index + 1])
index += 2

N = 1 << W

# Compute frequency g
g = [0] * N
for _ in range(H):
    row_str = data[index]
    val = int(row_str, 2)
    g[val] += 1
    index += 1

# Compute f_list
f_list = [min_hw_diff(d, W) for d in range(N)]

# Compute FWHT of f and g
Hf = fwht(f_list)
Hg = fwht(g)

# Pointwise product
P = [Hf[i] * Hg[i] for i in range(N)]

# Apply FWHT again
Hp = fwht(P)

# Compute convolution by dividing by N
conv = [hp // N for hp in Hp]

# Find the minimum sum
min_sum = min(conv)

# Output the result
print(min_sum)