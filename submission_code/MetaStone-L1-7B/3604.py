MOD = 10**9 + 7
max_n = 1000

# Precompute factorials and inverse factorials up to max_n
fact = [1] * (max_n + 1)
for i in range(1, max_n + 1):
    fact[i] = fact[i-1] * i % MOD

inv_fact = [1] * (max_n + 1)
inv_fact[max_n] = pow(fact[max_n], MOD - 2, MOD)
for i in range(max_n - 1, -1, -1):
    inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD

def numberOfWays(n, x, y):
    min_k = min(x, n)
    total = 0
    for k in range(1, min_k + 1):
        # Compute Stirling numbers of the second kind S(n, k)
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                if j == 1:
                    dp[i][j] = 1
                else:
                    dp[i][j] = (j * dp[i-1][j] + dp[i-1][j-1]) % MOD
        S_n_k = dp[n][k]
        
        # Compute combination C(x, k)
        if k > x:
            C_x_k = 0
        else:
            C_x_k = fact[x] * inv_fact[x - k] % MOD
            C_x_k = C_x_k * inv_fact[k] % MOD
        
        # Compute y^k
        y_pow_k = pow(y, k, MOD)
        
        # Compute term
        term = C_x_k * S_n_k % MOD
        term = term * fact[k] % MOD
        term = term * y_pow_k % MOD
        
        total = (total + term) % MOD
    
    return total