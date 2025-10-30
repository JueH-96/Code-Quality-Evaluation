class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        """
        Calculates the total number of ways an event with n performers, x stages, 
        and y possible scores can take place.

        The solution uses dynamic programming to first count the ways to form k bands,
        and then multiplies by the ways to score them.
        """
        MOD = 10**9 + 7

        # The number of bands `j` cannot exceed `n` (performers) or `x` (stages).
        limit = min(n, x)
        
        # dp[j] will store the number of ways to assign the current number of performers
        # to form exactly j bands. We use two arrays for space optimization.
        # `prev_dp` is for `i-1` performers, `curr_dp` is for `i` performers.
        
        # Initialize for 0 performers: 1 way to have 0 bands.
        prev_dp = [0] * (limit + 1)
        prev_dp[0] = 1

        # Iterate through each performer from 1 to n
        for i in range(1, n + 1):
            curr_dp = [0] * (limit + 1)
            # Iterate through the possible number of bands, j.
            # j cannot be more than the current number of performers `i` or the `limit`.
            for j in range(1, min(i, limit) + 1):
                # Case 1: i-th performer joins one of the j existing bands.
                term1 = (prev_dp[j] * j) % MOD
                
                # Case 2: i-th performer starts a new band on an unused stage.
                # If there were j-1 bands before, x - (j-1) stages were available.
                term2 = (prev_dp[j - 1] * (x - j + 1)) % MOD
                
                curr_dp[j] = (term1 + term2) % MOD
            prev_dp = curr_dp
    
        # After the loop, prev_dp holds the values for n performers.
        # prev_dp[j] is the number of ways to form j bands with n performers on x stages.
        final_assignments = prev_dp

        # Now, calculate the total ways by including the scoring possibilities.
        total_ways = 0
        y_pow_j = 1
        
        for j in range(1, limit + 1):
            # Calculate y^j efficiently for the current number of bands.
            y_pow_j = (y_pow_j * y) % MOD
            
            # Ways for j bands = (ways to form j bands) * (ways to score j bands)
            ways_for_j_bands = (final_assignments[j] * y_pow_j) % MOD
            
            total_ways = (total_ways + ways_for_j_bands) % MOD
            
        return total_ways