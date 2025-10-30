import sys
from collections import defaultdict

def mod_inverse(a, m=998244353):
    return pow(a, m-2, m)

def solve(N, A):
    MOD = 998244353
    dp = [0] * 11
    dp[0] = 1
    total_ways = 1
    
    for a in A:
        new_dp = [0] * 11
        for i in range(10, -1, -1):
            for j in range(1, a+1):
                if i + j <= 10:
                    new_dp[i+j] += dp[i]
                    new_dp[i+j] %= MOD
        dp = new_dp
        total_ways = (total_ways * a) % MOD
    
    valid_ways = sum(dp[10:]) % MOD
    return (valid_ways * mod_inverse(total_ways)) % MOD

input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))
print(solve(N, A))