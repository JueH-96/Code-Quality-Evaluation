class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        # dp[i] represents the maximum score using i elements
        dp = [-float('inf')] * 5
        dp[0] = 0
        
        for j in range(n):
            # Update in reverse order to avoid using updated values
            for i in range(min(4, j + 1), 0, -1):
                dp[i] = max(dp[i], dp[i-1] + a[i-1] * b[j])
        
        return dp[4]