class Solution:
    def waysToReachStair(self, k: int) -> int:
        if k == 0:
            return 2
        
        dp = [0] * (k + 1)
        dp[0] = 1
        dp[1] = 1
        
        for i in range(2, k + 1):
            dp[i] = dp[i - 1]
            j = 1
            while (1 << j) <= i:
                dp[i] += dp[i - (1 << j)]
                j += 1
        
        return dp[k]