def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

def count_divisible_by_exactly_one(x, n, m, l):
    # Count numbers <= x divisible by exactly one of n or m
    return x // n + x // m - 2 * (x // l)

N, M, K = map(int, input().split())

# Calculate LCM of N and M
L = lcm(N, M)

# Binary search for the K-th number
left, right = 1, 10**18  # Upper bound should be large enough

while left < right:
    mid = (left + right) // 2
    count = count_divisible_by_exactly_one(mid, N, M, L)
    
    if count < K:
        left = mid + 1
    else:
        right = mid

print(left)