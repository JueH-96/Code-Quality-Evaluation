class Solution:
    def waysToReachStair(self, k: int) -> int:
        mod = 10**9 + 7
        dp = [0] * (k + 1)
        dp[0] = 1
        dp[1] = 1
        for i in range(2, k + 1):
            j = 1
            while j <= i:
                dp[i] = (dp[i] + dp[i - j]) % mod
                j *= 2
        return dp[k]