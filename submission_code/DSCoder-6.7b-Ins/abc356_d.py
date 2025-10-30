def popcount(n):
    return bin(n).count('1')

def solve(N, M):
    result = 0
    for k in range(N+1):
        result = (result + popcount(k & M)) % 998244353
    return result

N, M = map(int, input().split())
print(solve(N, M))