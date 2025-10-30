class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        mod = 1000000007
        # dp[i][j] stands for the calculation result when faced with first i persons 
        # and first j stages 
        dp = [[0 for _ in range(x + 1)] for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(1, n + 1):
            for j in range(1, x + 1):
                # Don't allocate the i-th person
                dp[i][j] = dp[i - 1][j]
                # Allocate the i-th person
                dp[i][j] += dp[i - 1][j - 1]
                dp[i][j] %= mod
        ret = 0
        # For every possible band assign a score from 1 to y, 
        # use the power formula to speed up the process 
        for j in range(1, x + 1):
            ret += pow(y, j, mod) * dp[n][j]
            ret %= mod
        return ret