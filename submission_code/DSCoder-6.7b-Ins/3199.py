class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        dp = [0] * (n + 1)
        dp[0] = 1
        for i in range(1, n + 1):
            dp[i] = dp[i - 1] * 2
            if i - limit - 1 >= 0:
                dp[i] -= dp[i - limit - 1]
        return dp[n]