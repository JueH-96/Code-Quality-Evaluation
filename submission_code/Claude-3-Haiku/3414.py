class Solution:
    def waysToReachStair(self, k: int) -> int:
        if k == 0:
            return 2
        if k == 1:
            return 4
        
        dp = [0] * (k + 1)
        dp[0] = 2
        dp[1] = 4
        
        jump = 0
        for i in range(2, k + 1):
            dp[i] = dp[i - 1] + dp[max(0, i - 2 ** jump)]
            jump += 1
        
        return dp[k]