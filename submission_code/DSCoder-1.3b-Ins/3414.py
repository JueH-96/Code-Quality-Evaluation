class Solution:
    def waysToReachStair(self, k: int) -> int:
        dp = [0] * (k + 1)
        dp[0] = 1
        for i in range(1, k + 1):
            if i - 1 >= 0:
                dp[i] += dp[i - 1]
            if i - 2 >= 0:
                dp[i] += dp[i - 2]
        return dp[k]