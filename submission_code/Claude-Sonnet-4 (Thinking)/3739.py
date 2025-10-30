class Solution:
    def distanceSum(self, m: int, n: int, k: int) -> int:
        MOD = 10**9 + 7
        
        def mod_inverse(a, mod):
            return pow(a, mod - 2, mod)
        
        def combination_mod(n, r, mod):
            if r > n or r < 0:
                return 0
            if r == 0 or r == n:
                return 1
            
            numerator = 1
            denominator = 1
            for i in range(r):
                numerator = (numerator * (n - i)) % mod
                denominator = (denominator * (i + 1)) % mod
            
            return (numerator * mod_inverse(denominator, mod)) % mod
        
        total_cells = m * n
        if k < 2 or total_cells < k:
            return 0
        
        # Number of arrangements where any specific pair of positions both have pieces
        combinations = combination_mod(total_cells - 2, k - 2, MOD)
        
        # Calculate the sum of distances over all pairs of positions
        # Row contribution: n^2 * (m-1)*m*(m+1)/3 / 2
        # Column contribution: m^2 * (n-1)*n*(n+1)/3 / 2
        # Total = m*n/6 * [n*(m^2-1) + m*(n^2-1)]
        
        m_sq_minus_1 = (m * m - 1) % MOD
        n_sq_minus_1 = (n * n - 1) % MOD
        
        term1 = (n * m_sq_minus_1) % MOD
        term2 = (m * n_sq_minus_1) % MOD
        
        sum_term = (term1 + term2) % MOD
        
        # Multiply by m*n/6
        result = (m * n) % MOD
        result = (result * sum_term) % MOD
        result = (result * mod_inverse(6, MOD)) % MOD
        
        # Multiply by combinations
        result = (result * combinations) % MOD
        
        return result