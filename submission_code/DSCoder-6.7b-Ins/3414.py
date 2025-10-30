class Solution:
    def waysToReachStair(self, k: int) -> int:
        mod = 10**9 + 7
        dp = [0]*(k+1)
        dp[0] = dp[1] = 1
        for i in range(2, k+1):
            dp[i] = (dp[i-1] + dp[i-2]*2) % mod
        return dp[k]