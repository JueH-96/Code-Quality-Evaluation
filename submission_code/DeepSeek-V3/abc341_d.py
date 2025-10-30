# YOUR CODE HERE
import math

def count_divisible(x, n, m):
    # Count numbers <= x divisible by n or m but not both
    lcm_nm = (n * m) // math.gcd(n, m)
    count_n = x // n
    count_m = x // m
    count_lcm = x // lcm_nm
    return (count_n + count_m) - 2 * count_lcm

def find_kth_smallest(N, M, K):
    low = 1
    high = 10**18
    while low < high:
        mid = (low + high) // 2
        cnt = count_divisible(mid, N, M)
        if cnt < K:
            low = mid + 1
        else:
            high = mid
    return low

# Read input
N, M, K = map(int, input().split())

# Find the K-th smallest number
result = find_kth_smallest(N, M, K)

# Print the result
print(result)