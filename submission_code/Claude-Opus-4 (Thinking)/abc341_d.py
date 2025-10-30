def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

def count_exactly_one(X, N, M):
    L = lcm(N, M)
    return X // N + X // M - 2 * (X // L)

N, M, K = map(int, input().split())

# Binary search for the answer
left, right = 1, 10**20

while left < right:
    mid = (left + right) // 2
    if count_exactly_one(mid, N, M) < K:
        left = mid + 1
    else:
        right = mid

print(left)