class Solution:
    def distanceSum(self, m: int, n: int, k: int) -> int:
        MOD = 10**9 + 7
        
        def mod_inverse(a, mod):
            return pow(a, mod - 2, mod)
        
        def comb(n, k, mod):
            if k > n or k < 0:
                return 0
            if k == 0 or k == n:
                return 1
            
            # Optimize for k > n - k
            if k > n - k:
                k = n - k
            
            # Compute numerator
            num = 1
            for i in range(k):
                num = (num * (n - i)) % mod
            
            # Compute denominator
            den = 1
            for i in range(1, k + 1):
                den = (den * i) % mod
            
            # Return num / den mod p
            return (num * mod_inverse(den, mod)) % mod
        
        total_cells = m * n
        
        # Sum of Manhattan distances over all pairs of cells
        # We compute m * n * (m + n) * (m * n - 1) / 6
        # Since this is always an integer, we can use integer division
        sum_distances = m * n * (m + n) * (m * n - 1) // 6
        sum_distances %= MOD
        
        # Number of ways to choose the positions of k pieces such that two specific cells have pieces
        ways = comb(total_cells - 2, k - 2, MOD)
        
        # Total sum
        result = (ways * sum_distances) % MOD
        
        return result