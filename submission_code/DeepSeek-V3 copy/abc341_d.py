import math

def count_divisible_by_one(n, m, x):
    # Count numbers <= x divisible by n or m but not both
    count_n = x // n
    count_m = x // m
    lcm = (n * m) // math.gcd(n, m)
    count_both = x // lcm
    return count_n + count_m - 2 * count_both

def find_kth_smallest(n, m, k):
    low = 1
    high = 10**18
    while low < high:
        mid = (low + high) // 2
        cnt = count_divisible_by_one(n, m, mid)
        if cnt < k:
            low = mid + 1
        else:
            high = mid
    return low

# Read input
N, M, K = map(int, input().split())

# Find and print the K-th smallest number
print(find_kth_smallest(N, M, K))