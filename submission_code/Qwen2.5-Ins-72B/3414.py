class Solution:
    def waysToReachStair(self, k: int) -> int:
        if k == 0:
            return 2
        if k == 1:
            return 4
        
        # Initialize a DP array to store the number of ways to reach each stair
        dp = [0] * (k + 1)
        dp[0] = 2  # Base case: 2 ways to reach stair 0
        dp[1] = 4  # Base case: 4 ways to reach stair 1
        
        # Precompute the powers of 2 up to the maximum possible jump
        max_jump = 0
        while (1 << max_jump) <= k:
            max_jump += 1
        
        # Fill the DP array
        for i in range(2, k + 1):
            dp[i] = dp[i - 1]  # Using the down operation from stair i-1
            for jump in range(max_jump):
                if (1 << jump) == i:
                    dp[i] += dp[0]  # Using the up operation from stair 0
                elif (1 << jump) < i:
                    dp[i] += dp[i - (1 << jump)]  # Using the up operation from a lower stair
        
        return dp[k]