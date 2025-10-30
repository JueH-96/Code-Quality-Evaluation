class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        # dp[j] = max score choosing exactly j indices so far
        dp = [-float('inf')] * 5
        dp[0] = 0
        
        for i in range(len(b)):
            # We need to iterate j in reverse order to avoid using updated values
            for j in range(4, 0, -1):
                dp[j] = max(dp[j], dp[j-1] + a[j-1] * b[i])
        
        return dp[4]