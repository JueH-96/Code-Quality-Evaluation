def popcount(x):
    return bin(x).count('1')

def solve(N, M):
    MOD = 998244353
    result = 0
    for k in range(N + 1):
        result += popcount(k & M)
        result %= MOD
    return result

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
print(solve(N, M))