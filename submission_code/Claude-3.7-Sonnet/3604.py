class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7
        
        # Compute Stirling numbers of the second kind
        stirling = [[0] * (min(n, x) + 1) for _ in range(n + 1)]
        stirling[0][0] = 1
        
        for i in range(1, n + 1):
            for j in range(1, min(i, min(n, x)) + 1):
                stirling[i][j] = (j * stirling[i-1][j] + stirling[i-1][j-1]) % MOD
        
        total_ways = 0
        for k in range(1, min(n, x) + 1):
            # Compute P(x, k) - number of ways to select and order k stages from x
            permutation = 1
            for i in range(x, x - k, -1):
                permutation = (permutation * i) % MOD
            
            # Calculate ways for exactly k bands:
            # - permutation ways to choose k stages
            # - stirling[n][k] ways to partition n performers into k groups
            # - y^k ways to assign scores to the k bands
            ways_with_k_bands = (permutation * stirling[n][k] * pow(y, k, MOD)) % MOD
            total_ways = (total_ways + ways_with_k_bands) % MOD
            
        return total_ways