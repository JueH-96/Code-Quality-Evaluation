class Solution:
    def distanceSum(self, m: int, n: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # For each valid arrangement, we need to calculate the sum of Manhattan distances
        # between every pair of pieces. The key insight is to calculate this combinatorially.
        
        # Step 1: Calculate the number of arrangements with pieces at two specific cells
        # This is (m*n-2 choose k-2)
        if k == 2:
            combinations = 1  # Special case for k=2
        else:
            numerator = 1
            denominator = 1
            for i in range(k-2):
                numerator = (numerator * (m*n-2-i)) % MOD
                denominator = (denominator * (i+1)) % MOD
            inverse_denominator = pow(denominator, MOD-2, MOD)
            combinations = (numerator * inverse_denominator) % MOD
        
        # Step 2: Calculate the sum of Manhattan distances over all pairs of distinct positions
        # The formula for the sum of Manhattan distances over all pairs in a grid is:
        # [n²(m-1)m(m+1) + m²(n-1)n(n+1)]/6
        
        n_squared = pow(n, 2, MOD)
        m_squared = pow(m, 2, MOD)
        
        term1 = (n_squared * ((m-1) * m % MOD) % MOD * (m+1)) % MOD
        term2 = (m_squared * ((n-1) * n % MOD) % MOD * (n+1)) % MOD
        
        sum_distances = (term1 + term2) % MOD
        inv_6 = pow(6, MOD-2, MOD)  # Modular inverse of 6
        sum_distances = (sum_distances * inv_6) % MOD
        
        # Final answer: multiply the number of arrangements with the sum of distances
        return (combinations * sum_distances) % MOD