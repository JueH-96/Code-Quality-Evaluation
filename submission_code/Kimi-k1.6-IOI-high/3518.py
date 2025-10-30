class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        dp = [-float('inf')] * 5
        dp[0] = 0
        for x in b:
            for j in range(4, 0, -1):
                dp[j] = max(dp[j], dp[j-1] + a[j-1] * x)
        return dp[4]