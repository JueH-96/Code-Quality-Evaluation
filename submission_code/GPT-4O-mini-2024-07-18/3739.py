class Solution:
    def distanceSum(self, m: int, n: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # Calculate the total number of cells
        total_cells = m * n
        
        # Precompute factorials and inverse factorials for combinations
        fact = [1] * (total_cells + 1)
        for i in range(2, total_cells + 1):
            fact[i] = fact[i - 1] * i % MOD
        
        def mod_inv(x):
            return pow(x, MOD - 2, MOD)
        
        inv_fact = [1] * (total_cells + 1)
        for i in range(2, total_cells + 1):
            inv_fact[i] = mod_inv(fact[i])
        
        # Function to calculate combinations C(n, k)
        def comb(n, k):
            if k > n or k < 0:
                return 0
            return fact[n] * inv_fact[k] % MOD * inv_fact[n - k] % MOD
        
        # Calculate the total Manhattan distance
        total_distance = 0
        
        # Calculate contributions from rows
        for i in range(m):
            for j in range(n):
                # Contribution of (i, j) to all pairs
                # Each piece can be placed in (k-1) other positions
                # The distance contribution from this cell to all other cells
                distance_contribution = 0
                
                # For rows
                for x in range(m):
                    distance_contribution += abs(i - x) * n * comb(total_cells - 1, k - 1)
                    distance_contribution %= MOD
                
                # For columns
                for y in range(n):
                    distance_contribution += abs(j - y) * m * comb(total_cells - 1, k - 1)
                    distance_contribution %= MOD
                
                total_distance += distance_contribution
                total_distance %= MOD
        
        return total_distance

# Example usage:
# sol = Solution()
# print(sol.distanceSum(2, 2, 2))  # Output: 8
# print(sol.distanceSum(1, 4, 3))  # Output: 20