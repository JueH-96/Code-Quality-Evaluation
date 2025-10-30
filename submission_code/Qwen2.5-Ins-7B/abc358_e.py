# MOD = 998244353
from math import comb

def count_strings(K, C):
    MOD = 998244353
    dp = [1] + [0] * K
    for c in C:
        for i in range(K, -1, -1):
            if dp[i] > 0:
                for j in range(1, min(c + 1, i + 1)):
                    dp[i + j] = (dp[i + j] + dp[i] * comb(i, j)) % MOD
    return dp[K]

K = int(input())
C = list(map(int, input().split()))
print(count_strings(K, C))