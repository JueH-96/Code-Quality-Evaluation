# YOUR CODE HERE

def popcount(x):
    return bin(x).count('1')

def solve(N, M):
    mod = 998244353
    result = 0
    for k in range(N+1):
        result += popcount(k & M)
        result %= mod
    return result

N, M = map(int, input().split())
print(solve(N, M))