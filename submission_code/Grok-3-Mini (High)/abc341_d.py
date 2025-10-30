import math

# Read input from stdin
N, M, K = map(int, input().split())

# Compute GCD and LCM
gcd_nm = math.gcd(N, M)
L = (N * M) // gcd_nm

# Function to count numbers <= X divisible by exactly one of N or M
def count_up_to(X):
    return (X // N) + (X // M) - 2 * (X // L)

# Binary search to find the smallest X such that count_up_to(X) >= K
low = 1
high = 2**60  # Upper bound large enough to cover maximum possible X

while low < high:
    mid = (low + high) // 2
    if count_up_to(mid) >= K:
        high = mid
    else:
        low = mid + 1

# Output the result
print(low)