import sys
from collections import defaultdict

MOD = 998244353

def modinv(x, mod=MOD):
    return pow(x, mod-2, mod)

def solve(n, m, conditions):
    # Initialize the DP table
    dp = [0] * (n + 1)
    dp[0] = 1
    
    # Precompute factorials and their inverses
    fact = [1] * (n + 1)
    inv_fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i-1] * i % MOD
        inv_fact[i] = modinv(fact[i])
    
    # Process conditions
    constraints = defaultdict(list)
    for l, r, x in conditions:
        constraints[r].append((l, x))
    
    for i in range(1, n + 1):
        dp[i] = dp[i-1] * i % MOD
        for l, x in constraints[i]:
            if l <= x <= i:
                dp[i] -= dp[l-1] * fact[i-l] % MOD * inv_fact[i-l-1] % MOD
                dp[i] %= MOD
    
    return dp[n]

# Read input
input = sys.stdin.read
data = input().split()
n = int(data[0])
m = int(data[1])
conditions = [(int(data[2*i+2]), int(data[2*i+3]), int(data[2*i+4])) for i in range(m)]

# Solve and print the result
print(solve(n, m, conditions))