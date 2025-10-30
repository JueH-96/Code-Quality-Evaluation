class Solution:
    def waysToReachStair(self, k: int) -> int:
        MOD = 10**9 + 7
        dp = [0]*(k+1)
        dp[0] = 2
        for i in range(1, k+1):
            dp[i] = dp[i-1]
            if i >= 2:
                dp[i] += dp[i-2]
            dp[i] %= MOD
        return dp[k]