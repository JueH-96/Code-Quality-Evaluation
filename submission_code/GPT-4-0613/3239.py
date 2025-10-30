class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        dp = [0] + [float('inf')] * 20000
        for i in range(1, x + 1):
            dp[i] = min(dp[i], dp[i - 1] + 1)
            if i % 5 == 0:
                dp[i // 5] = min(dp[i // 5], dp[i] + 1)
            if i % 11 == 0:
                dp[i // 11] = min(dp[i // 11], dp[i] + 1)
        for i in range(x + 1, 20000):
            dp[i] = min(dp[i], dp[i - 1] + 1)
        return dp[y]