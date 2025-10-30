class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        
        # If n is less than 4, it's impossible to form "leet"
        if n < 4:
            return 0
        
        # Total number of strings of length n
        total_strings = pow(26, n, MOD)
        
        # Total number of strings of length n that do not contain "leet"
        # We need to subtract the number of strings that do not have "leet" as a substring
        # This is a complex combinatorial problem, but we can use inclusion-exclusion principle
        # and dynamic programming to solve it efficiently.
        
        # We need to find the number of strings of length n that do not contain "leet"
        # as a substring. This is equivalent to counting the number of ways to fill
        # a string of length n with 26 letters such that "leet" never appears as a substring.
        
        # dp[i] will store the number of strings of length i that do not contain "leet"
        dp = [0] * (n + 1)
        
        # Base cases
        dp[0] = 1  # There's one empty string
        dp[1] = 26  # 26 single character strings
        dp[2] = 26 * 26  # 26^2 two character strings
        dp[3] = 26 * 26 * 26  # 26^3 three character strings
        
        # For strings of length 4 or more, we need to ensure "leet" is not a substring
        for i in range(4, n + 1):
            # We can append any of the 26 letters to a valid string of length i-1
            dp[i] = (dp[i - 1] * 26) % MOD
            
            # Subtract the number of strings that end with "leet"
            # If a string ends with "leet", the first i-4 characters can be anything
            dp[i] = (dp[i] - dp[i - 4]) % MOD
        
        # The number of strings that do not contain "leet" as a substring
        no_leet_strings = dp[n]
        
        # The number of good strings is total strings minus those without "leet"
        good_strings = (total_strings - no_leet_strings) % MOD
        
        return good_strings