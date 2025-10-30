class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7
        
        # Compute Stirling numbers of the second kind
        S = [[0] * (n + 1) for _ in range(n + 1)]
        S[0][0] = 1
        for i in range(1, n + 1):
            for j in range(1, i + 1):
                S[i][j] = (j * S[i-1][j] + S[i-1][j-1]) % MOD
        
        # Compute factorials
        fact = [1] * (x + 1)
        for i in range(1, x + 1):
            fact[i] = fact[i-1] * i % MOD
        
        # Compute inverse factorials for binomial coefficients
        inv_fact = [1] * (x + 1)
        inv_fact[x] = pow(fact[x], MOD - 2, MOD)
        for i in range(x - 1, -1, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
        
        def C(n, k):
            if k > n or k < 0:
                return 0
            return fact[n] * inv_fact[k] % MOD * inv_fact[n - k] % MOD
        
        result = 0
        for k in range(1, min(n, x) + 1):
            ways = C(x, k) * fact[k] % MOD * S[n][k] % MOD * pow(y, k, MOD) % MOD
            result = (result + ways) % MOD
        
        return result