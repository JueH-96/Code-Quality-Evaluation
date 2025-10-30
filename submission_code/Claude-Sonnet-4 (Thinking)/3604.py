class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7
        
        # Precompute factorials and their modular inverses
        fact = [1] * 1001
        for i in range(1, 1001):
            fact[i] = fact[i-1] * i % MOD
        
        inv_fact = [1] * 1001
        inv_fact[1000] = pow(fact[1000], MOD - 2, MOD)
        for i in range(999, -1, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
        
        # Function to compute combination C(n, k)
        def comb(n, k):
            if k > n or k < 0:
                return 0
            return fact[n] * inv_fact[k] % MOD * inv_fact[n - k] % MOD
        
        # Compute Stirling numbers of the second kind using DP
        # S(n,k) = k*S(n-1,k) + S(n-1,k-1)
        max_k = min(n, x)
        S = [[0] * (max_k + 1) for _ in range(n + 1)]
        S[0][0] = 1
        
        for i in range(1, n + 1):
            for j in range(1, min(i, max_k) + 1):
                S[i][j] = (j * S[i-1][j] + S[i-1][j-1]) % MOD
        
        total = 0
        for k in range(1, max_k + 1):
            ways = comb(x, k) * S[n][k] % MOD
            ways = ways * fact[k] % MOD
            ways = ways * pow(y, k, MOD) % MOD
            total = (total + ways) % MOD
        
        return total