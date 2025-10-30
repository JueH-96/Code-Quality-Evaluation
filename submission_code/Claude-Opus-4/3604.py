class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7
        
        # Precompute Stirling numbers of the second kind S(n, k)
        # S(n, k) = number of ways to partition n objects into k non-empty subsets
        S = [[0] * (min(n, x) + 1) for _ in range(n + 1)]
        S[0][0] = 1
        
        for i in range(1, n + 1):
            for j in range(1, min(i, x) + 1):
                S[i][j] = (j * S[i-1][j] + S[i-1][j-1]) % MOD
        
        # Precompute factorials
        fact = [1] * (x + 1)
        for i in range(1, x + 1):
            fact[i] = (fact[i-1] * i) % MOD
        
        # Precompute inverse factorials for combinations
        inv_fact = [1] * (x + 1)
        inv_fact[x] = pow(fact[x], MOD - 2, MOD)
        for i in range(x - 1, -1, -1):
            inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD
        
        # Precompute powers of y
        y_pow = [1] * (min(n, x) + 1)
        for i in range(1, min(n, x) + 1):
            y_pow[i] = (y_pow[i-1] * y) % MOD
        
        result = 0
        
        # For each possible number of non-empty stages
        for k in range(1, min(n, x) + 1):
            # Ways to partition n performers into k non-empty groups
            partitions = S[n][k]
            
            # Ways to assign k groups to x stages
            # This is P(x, k) = x! / (x-k)!
            assignments = (fact[x] * inv_fact[x - k]) % MOD
            
            # Ways to assign scores to k bands
            scores = y_pow[k]
            
            # Add to result
            result = (result + partitions * assignments % MOD * scores % MOD) % MOD
        
        return result