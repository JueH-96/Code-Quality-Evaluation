class Solution:
    def waysToReachStair(self, k: int) -> int:
        # Base case
        if k == 0:
            return 2
        if k == 1:
            return 4
        
        # Dynamic programming approach
        dp = [0] * (k + 1)
        dp[0] = 2  # There are 2 ways to stay at stair 0
        dp[1] = 4  # There are 4 ways to stay at stair 1
        
        # Calculate the number of ways for each stair from 2 to k
        for i in range(2, k + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        
        return dp[k]