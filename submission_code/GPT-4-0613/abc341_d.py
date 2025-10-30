import math

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

N, M, K = map(int, input().split())

l = lcm(N, M)
left = 1
right = K * min(N, M) + 1

while right - left > 1:
    mid = (left + right) // 2
    if mid // N + mid // M - mid // l < K:
        left = mid
    else:
        right = mid

print(left)