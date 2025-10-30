from math import gcd

def lcm(a, b):
    return a * b // gcd(a, b)

n, m, k = map(int, input().split())
l = lcm(n, m)

base = n + m - l // n - l // m

# if k is greater than base, calculate k modulo base-th term
if k > base:
    k = k % base

left, right = 1, 10**18
while left < right:
    mid = (left + right) // 2
    cnt = mid // n + mid // m - mid // l
    if cnt < k:
        left = mid + 1
    else:
        right = mid

print(left)