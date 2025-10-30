import sys
from collections import defaultdict

def mod_pow(base, exp, mod):
    result = 1
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp //= 2
    return result

def mod_inv(x, mod):
    return mod_pow(x, mod - 2, mod)

def solve(K, C):
    mod = 998244353
    fact = [1] * (K + 1)
    inv_fact = [1] * (K + 1)
    for i in range(2, K + 1):
        fact[i] = (fact[i - 1] * i) % mod
        inv_fact[i] = mod_inv(fact[i], mod)
    
    dp = [0] * (K + 1)
    dp[0] = 1
    for i in range(26):
        for j in range(K, 0, -1):
            for k in range(1, min(j, C[i]) + 1):
                dp[j] = (dp[j] + dp[j - k] * fact[j] * inv_fact[j - k] * inv_fact[k]) % mod
    
    result = sum(dp[1:]) % mod
    return result

input = sys.stdin.read
data = input().split()
K = int(data[0])
C = list(map(int, data[1:]))
print(solve(K, C))