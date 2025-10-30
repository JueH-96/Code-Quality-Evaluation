def solve():
    MOD = 998244353
    N = int(input())
    
    # Compute number of digits for each number
    d = [0] * (N + 1)
    for i in range(1, N + 1):
        d[i] = len(str(i))
    
    # Precompute powers of 10
    pow10 = [0] * (N + 1)
    for i in range(1, N + 1):
        pow10[i] = pow(10, d[i], MOD)
    
    # Compute G[j] using DP with rolling array
    # G[j] = sum over all j-subsets T of {1, ..., N}: 10^{sum_{t in T} d[t]}
    prev = [0] * (N + 1)
    prev[0] = 1
    
    for m in range(1, N + 1):
        curr = [0] * (N + 1)
        for j in range(min(m + 1, N + 1)):
            curr[j] = prev[j]
            if j > 0:
                curr[j] = (curr[j] + prev[j-1] * pow10[m]) % MOD
        prev = curr
    
    G = prev
    
    # Compute the answer
    ans = 0
    
    # Precompute factorials
    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = fact[i-1] * i % MOD
    
    for k in range(1, N + 1):
        # Compute H[k][j] for all j
        # H[k][j] = sum over all j-subsets T of {1, ..., N} \ {k}: 10^{sum_{t in T} d[t]}
        H = [0] * (N + 1)
        H[0] = 1
        for j in range(1, N + 1):
            H[j] = (G[j] - pow10[k] * H[j-1] % MOD + MOD) % MOD
        
        # Add contribution of k
        for i in range(1, N + 1):
            contrib = k * fact[i-1] % MOD * fact[N-i] % MOD * H[N-i] % MOD
            ans = (ans + contrib) % MOD
    
    return ans

print(solve())