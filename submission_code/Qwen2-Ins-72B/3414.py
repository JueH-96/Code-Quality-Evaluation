class Solution:
    def waysToReachStair(self, k: int) -> int:
        MOD = 10**9 + 7
        if k == 0:
            return 2
        if k == 1:
            return 4
        dp = [0] * (k + 1)
        dp[0], dp[1] = 2, 4
        for i in range(2, k + 1):
            dp[i] = dp[i - 1] * 2 % MOD
            j = 1
            while (1 << j) <= i:
                dp[i] = (dp[i] + dp[i - (1 << j)] * 2) % MOD
                j += 1
        return dp[k]