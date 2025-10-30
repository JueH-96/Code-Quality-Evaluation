class Solution:
    def waysToReachStair(self, k: int) -> int:
        if k == 0:
            return 2
        
        # dp[i] will store the number of ways to reach stair i
        dp = [0] * (k + 1)
        dp[0] = 1  # 1 way to stay at stair 0 (do nothing)
        dp[1] = 1  # 1 way to reach stair 1 (start there)
        
        for i in range(2, k + 1):
            dp[i] = dp[i - 1]  # coming from stair i-1
            jump = 0
            while (i - (1 << jump)) >= 0:
                dp[i] += dp[i - (1 << jump)]
                jump += 1
        
        return dp[k] * 2  # multiply by 2 for the two ways to reach stair 0