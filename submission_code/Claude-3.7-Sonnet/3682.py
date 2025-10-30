class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # Base case: if we need more equal pairs than possible
        if k >= n:
            return 0
            
        # prev_dp[j] = number of arrays of current length with j equal adjacent pairs
        prev_dp = [0] * (k+1)
        
        # Arrays of length 1 have 0 adjacent pairs
        prev_dp[0] = m
        
        for i in range(1, n):
            curr_dp = [0] * (k+1)
            
            for j in range(min(i+1, k+1)):
                # Case 1: Current element is same as previous element
                # This creates a new equal pair, so we use prev state with (j-1) pairs
                if j > 0:
                    curr_dp[j] = (curr_dp[j] + prev_dp[j-1]) % MOD
                
                # Case 2: Current element is different from previous element
                # This doesn't create a new equal pair, so we use prev state with j pairs
                # We have (m-1) choices for the different element
                curr_dp[j] = (curr_dp[j] + (m-1) * prev_dp[j]) % MOD
            
            prev_dp = curr_dp
        
        return prev_dp[k]