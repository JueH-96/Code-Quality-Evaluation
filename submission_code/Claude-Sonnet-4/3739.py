class Solution:
    def distanceSum(self, m: int, n: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # Precompute factorials and inverse factorials for combinations
        max_val = m * n
        fact = [1] * (max_val + 1)
        for i in range(1, max_val + 1):
            fact[i] = (fact[i-1] * i) % MOD
        
        def pow_mod(base, exp, mod):
            result = 1
            base %= mod
            while exp > 0:
                if exp & 1:
                    result = (result * base) % mod
                base = (base * base) % mod
                exp >>= 1
            return result
        
        inv_fact = [1] * (max_val + 1)
        inv_fact[max_val] = pow_mod(fact[max_val], MOD - 2, MOD)
        for i in range(max_val - 1, -1, -1):
            inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD
        
        def comb(n, r):
            if r < 0 or r > n:
                return 0
            return (fact[n] * inv_fact[r] % MOD) * inv_fact[n - r] % MOD
        
        # Number of ways to choose k-2 pieces from remaining m*n-2 cells
        arrangements = comb(m * n - 2, k - 2)
        
        result = 0
        
        # Calculate contribution from row differences
        for r1 in range(m):
            for r2 in range(m):
                if r1 != r2:
                    row_diff = abs(r1 - r2)
                    # n^2 pairs of cells between these two rows
                    contribution = (row_diff * n * n % MOD * arrangements) % MOD
                    result = (result + contribution) % MOD
        
        # Calculate contribution from column differences  
        for c1 in range(n):
            for c2 in range(n):
                if c1 != c2:
                    col_diff = abs(c1 - c2)
                    # m^2 pairs of cells between these two columns
                    contribution = (col_diff * m * m % MOD * arrangements) % MOD
                    result = (result + contribution) % MOD
        
        return result