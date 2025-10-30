import math

N, M, K = map(int, input().split())
gcd = math.gcd(N, M)
lcm = N * M // gcd

def count(x):
    return x // N + x // M - 2 * (x // lcm)

low = 1
high = 1

while count(high) < K:
    high *= 2

while low < high:
    mid = (low + high) // 2
    if count(mid) >= K:
        high = mid
    else:
        low = mid + 1

print(low)