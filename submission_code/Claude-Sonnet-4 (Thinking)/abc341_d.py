def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

def count_divisible_by_exactly_one(x, N, M, lcm_NM):
    return x // N + x // M - 2 * (x // lcm_NM)

N, M, K = map(int, input().split())

lcm_NM = lcm(N, M)

left, right = 1, 10**18

while left < right:
    mid = (left + right) // 2
    if count_divisible_by_exactly_one(mid, N, M, lcm_NM) >= K:
        right = mid
    else:
        left = mid + 1

print(left)