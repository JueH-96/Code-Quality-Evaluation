import math

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

def count_divisible_by_exactly_one(x, n, m, lcm_nm):
    # Count numbers <= x divisible by exactly one of n or m
    count_n = x // n
    count_m = x // m
    count_both = x // lcm_nm
    return count_n + count_m - 2 * count_both

N, M, K = map(int, input().split())

lcm_nm = lcm(N, M)

# Binary search for the answer
left, right = 1, K * min(N, M)

while left < right:
    mid = (left + right) // 2
    count = count_divisible_by_exactly_one(mid, N, M, lcm_nm)
    
    if count >= K:
        right = mid
    else:
        left = mid + 1

# Find the exact K-th number
result = left
while True:
    # Check if result is divisible by exactly one of N or M
    div_by_n = (result % N == 0)
    div_by_m = (result % M == 0)
    
    if div_by_n and not div_by_m:
        break
    elif div_by_m and not div_by_n:
        break
    else:
        result += 1

print(result)