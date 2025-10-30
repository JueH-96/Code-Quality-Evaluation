class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        # We want to pick 4 indices i0 < i1 < i2 < i3 in b such that
        # our score is a[0]*b[i0] + a[1]*b[i1] + a[2]*b[i2] + a[3]*b[i3].
        
        # We can use a running DP approach for k=0..4 picks:
        # dp[k] will represent the maximum score we can achieve with k picks
        # while iterating through b from left to right.
        # Initialize dp for k=0..4
        dp = [float('-inf')] * 5
        dp[0] = 0  # With 0 picks, the score is 0
        
        for val in b:
            # Iterate backwards from 4 down to 1 to ensure we don't overwrite
            # results we still need to use for the next states.
            for k in range(4, 0, -1):
                dp[k] = max(dp[k], dp[k-1] + a[k-1] * val)
        
        # The answer is dp[4], corresponding to picking exactly 4 elements.
        return dp[4]