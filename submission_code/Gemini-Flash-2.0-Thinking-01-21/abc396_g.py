import sys

# Function to calculate popcount (number of set bits)
# Using Python's built-in method for efficiency
def popcount(n):
    return bin(n).count('1')
    # Alternative manual implementation:
    # count = 0
    # while n > 0:
    #     n &= (n - 1)
    #     count += 1
    # return count

# Fast Walsh-Hadamard Transform (FWHT) for XOR convolution (sum/difference variant)
# This version performs the transform in-place.
def fwht(a):
    n = len(a)
    if n == 0:
        return []

    # Ensure n is a power of 2
    # This is guaranteed by N = 1 << W where W >= 1.
    # assert n & (n - 1) == 0

    length = 1
    while length < n:
        for i in range(0, n, length << 1):
            for j in range(length):
                u = a[i + j]
                v = a[i + length + j]
                a[i + j] = u + v
                a[i + length + j] = u - v
        length <<= 1
    return a

# Read input
H, W = map(int, sys.stdin.readline().split())
initial_row_masks = []
for _ in range(H):
    row_str = sys.stdin.readline().strip()
    mask = int(row_str, 2)
    initial_row_masks.append(mask)

N = 1 << W # 2^W

# Array A: counts of initial row masks
# Initialize with zeros, size 2^W
A = [0] * N
for mask in initial_row_masks:
    A[mask] += 1

# Array B: g(mask) = min(popcount(mask), W - popcount(mask))
# Initialize with zeros, size 2^W
B = [0] * N
for mask in range(N):
    B[mask] = min(popcount(mask), W - popcount(mask))

# Compute FWHT of A and B
# Use list() to pass a copy so the original A and B are not modified
hat_A = fwht(list(A))
hat_B = fwht(list(B))

# Compute element-wise product hat_S = hat_A .* hat_B
# hat_S[k] = hat_A[k] * hat_B[k]
hat_S = [hat_A[i] * hat_B[i] for i in range(N)]

# Compute Inverse FWHT of hat_S
# The sum we want is S[c] = sum_m A[m] B[m ^ c], which is the XOR convolution (A * B)[c]
# The relationship is S = (1/N) * FWHT(hat_S) using the sum/difference transform
S_unscaled = fwht(list(hat_S))

# Scale the results to get the actual sums
# The problem guarantees integer sums.
# This implies S_unscaled[c] should be divisible by N for all c.
# Using integer division is appropriate here.
S = [s // N for s in S_unscaled]

# The minimum value in S is the minimum total sum over all possible column flip combinations
min_total_sum = float('inf')
for sum_val in S:
    min_total_sum = min(min_total_sum, sum_val)

print(min_total_sum)