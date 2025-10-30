import math

n, m, k = map(int, input().split())

gcd = math.gcd(n, m)
lcm = (n * m) // gcd

def count_valid(x):
    return (x // n) + (x // m) - 2 * (x // lcm)

# Find the lower and upper bounds for binary search
low = 1
high = 1

# Expand the upper bound until count >= K
while count_valid(high) < k:
    high *= 2

# Binary search to find the minimal X where count_valid(X) >= K
while low < high:
    mid = (low + high) // 2
    current = count_valid(mid)
    if current < k:
        low = mid + 1
    else:
        high = mid

print(low)