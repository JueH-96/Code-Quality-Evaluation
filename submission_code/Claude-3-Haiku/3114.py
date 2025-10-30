class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        dp = [0] * n
        
        # Build the mountain from the left
        for i in range(n):
            dp[i] = min(maxHeights[i], i + 1)
        
        # Build the mountain from the right
        for i in range(n - 2, -1, -1):
            dp[i] = min(dp[i], min(maxHeights[i], n - i))
        
        return sum(dp)