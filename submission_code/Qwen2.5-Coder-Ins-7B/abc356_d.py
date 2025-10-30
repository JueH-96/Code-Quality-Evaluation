MOD = 998244353

def popcount(x):
    return bin(x).count('1')

def solve(N, M):
    result = 0
    for k in range(N + 1):
        result += popcount(k & M)
    return result % MOD

# Read input
N, M = map(int, input().split())

# Solve and print the result
print(solve(N, M))