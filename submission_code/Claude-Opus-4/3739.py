class Solution:
    def distanceSum(self, m: int, n: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # Precompute factorials
        max_val = m * n + 1
        fact = [1] * max_val
        for i in range(1, max_val):
            fact[i] = fact[i-1] * i % MOD
        
        # Function to compute modular inverse
        def mod_inverse(x):
            return pow(x, MOD - 2, MOD)
        
        # Function to compute C(n, k) % MOD
        def comb(n, k):
            if k > n or k < 0:
                return 0
            return fact[n] * mod_inverse(fact[k]) % MOD * mod_inverse(fact[n-k]) % MOD
        
        # Calculate sum of distances
        result = 0
        
        # For row distances
        for d in range(1, m):
            # Count pairs of cells that are d rows apart
            count_pairs = (m - d) * n * n
            # Each such pair appears in C(m*n-2, k-2) arrangements
            result = (result + d * count_pairs * comb(m * n - 2, k - 2)) % MOD
        
        # For column distances
        for d in range(1, n):
            # Count pairs of cells that are d columns apart
            count_pairs = (n - d) * m * m
            # Each such pair appears in C(m*n-2, k-2) arrangements
            result = (result + d * count_pairs * comb(m * n - 2, k - 2)) % MOD
        
        return result