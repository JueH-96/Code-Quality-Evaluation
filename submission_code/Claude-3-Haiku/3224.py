class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        dp = [0] * n
        
        # Initialize the first and last healthy children
        for i in range(n):
            if i not in sick:
                dp[i] = 1
        
        # Iterate through the sick children
        for i in sick:
            # If the left neighbor is healthy, infect it
            if i > 0 and i-1 not in sick:
                dp[i-1] = (dp[i-1] + dp[i]) % MOD
            # If the right neighbor is healthy, infect it
            if i < n-1 and i+1 not in sick:
                dp[i+1] = (dp[i+1] + dp[i]) % MOD
        
        # Sum up the number of possible infection sequences
        return sum(dp) % MOD