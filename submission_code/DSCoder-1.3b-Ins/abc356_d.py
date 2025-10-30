import sys

def popcount(n):
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count

def solve(N, M):
    result = 0
    for k in range(N + 1):
        result += popcount(k & M)
        result %= 998244353
    return result

N, M = map(int, sys.stdin.readline().split())
print(solve(N, M))