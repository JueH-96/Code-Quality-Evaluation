class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        # We will use a dynamic programming approach where dp[j] represents
        # the maximum score we can achieve by picking j elements so far.
        # We iterate through elements of b in order, and for each b[i],
        # we update dp[j] from right to left. This ensures that when we update
        # dp[j], we are using the dp[j-1] from the previous state (previous i),
        # ensuring the correct choice of indices i0 < i1 < i2 < i3.
        
        # Initialize dp array: dp[0] = 0, and dp[j] = -inf for j > 0
        dp = [float('-inf')] * 5
        dp[0] = 0
        
        for val in b:
            # Update dp[j] from right to left to avoid overwriting needed values
            for j in range(4, 0, -1):
                dp[j] = max(dp[j], dp[j-1] + a[j-1] * val)
        
        return dp[4]