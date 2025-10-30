class Solution:
    def waysToReachStair(self, k: int) -> int:
        if k == 0:
            return 2
        elif k == 1:
            return 4
        
        # Dynamic programming approach to count ways to reach stair k
        # dp[i] will store the number of ways to reach stair i
        dp = [0] * (k + 1)
        dp[0] = 2  # Base case: 2 ways to stay at 0 (as per example)
        dp[1] = 4  # Base case: 4 ways to reach 1 (as per example)
        
        # To reach any stair i > 1, we can come from any stair j where j < i
        # and j can be reached by a power of 2 jump from some stair m (m < j)
        # We also need to consider the ways to reach back to 0 and then to i
        for i in range(2, k + 1):
            # Start with ways to reach i directly from 0 or by bouncing back to 0
            dp[i] = dp[0]  # All ways that reach 0 can reach i by a direct jump if possible
            
            # Check all powers of 2 jumps that could land on i
            jump = 0
            while (1 << jump) <= i:
                from_stair = i - (1 << jump)
                if from_stair >= 0:
                    dp[i] += dp[from_stair]
                jump += 1
        
        return dp[k]