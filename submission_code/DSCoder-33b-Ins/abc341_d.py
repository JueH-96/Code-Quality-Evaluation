# YOUR CODE HERE
N, M, K = map(int, input().split())

def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)

def lcm(a, b):
    return a * b // gcd(a, b)

low = 1
high = 10**18

while low < high:
    mid = (low + high) // 2
    count = mid // N + mid // M - mid // lcm(N, M)
    if count < K:
        low = mid + 1
    else:
        high = mid

print(low)