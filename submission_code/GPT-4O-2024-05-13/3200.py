class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        
        if n < 4:
            return 0
        
        # Total number of strings of length n
        total_strings = pow(26, n, MOD)
        
        # Total number of strings of length n that do not contain "leet"
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for i in range(1, n + 1):
            dp[i] = (dp[i - 1] * 25) % MOD
            if i >= 4:
                dp[i] = (dp[i] - dp[i - 4] * 24) % MOD
        
        non_good_strings = dp[n]
        
        # Good strings are total strings minus non-good strings
        good_strings = (total_strings - non_good_strings + MOD) % MOD
        
        return good_strings