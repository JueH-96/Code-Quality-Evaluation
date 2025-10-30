import math

n, m, k = map(int, input().split())

gcd = math.gcd(n, m)
lcm = (n * m) // gcd
max_val = max(n, m)

low = 1
high = max_val * k

# Ensure high is large enough
while True:
    cnt = (high // n) + (high // m) - 2 * (high // lcm)
    if cnt >= k:
        break
    high *= 2

# Binary search to find the minimal X
while low < high:
    mid = (low + high) // 2
    cnt = (mid // n) + (mid // m) - 2 * (mid // lcm)
    if cnt >= k:
        high = mid
    else:
        low = mid + 1

print(low)