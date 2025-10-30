class Solution:
    def waysToReachStair(self, k: int) -> int:
        MOD = 10**9 + 7
        
        # dp[i] will store the number of ways to reach stair i
        dp = [0] * (k + 2)
        dp[1] = 1  # Starting point
        
        for i in range(1, k + 1):
            # If we are at stair i, we can go down to stair i-1
            if i > 0:
                dp[i - 1] = (dp[i - 1] + dp[i]) % MOD
            
            # If we are at stair i, we can go up to stair i + 2^jump
            jump = 0
            while i + (1 << jump) <= k + 1:
                dp[i + (1 << jump)] = (dp[i + (1 << jump)] + dp[i]) % MOD
                jump += 1
        
        # We need to count all ways to reach stair k from any stair
        result = 0
        for i in range(k + 1):
            if i + 1 <= k:
                result = (result + dp[i]) % MOD
            if i - 1 >= 0:
                result = (result + dp[i + 1]) % MOD
        
        return result