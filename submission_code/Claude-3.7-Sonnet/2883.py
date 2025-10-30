class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        n = len(s)
        
        # Generate powers of 5 in binary representation
        powers_of_5 = {bin(5 ** i)[2:] for i in range(7)}  # 5^6 < 2^15, so 7 is enough
        
        # Initialize dp array where dp[i] represents the minimum number
        # of beautiful substrings needed to partition s[0:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            for j in range(i):
                # Check if substring s[j:i] is beautiful:
                # 1. No leading zeros (s[j] != '0')
                # 2. It's a binary representation of a power of 5
                if s[j] != '0' and s[j:i] in powers_of_5:
                    dp[i] = min(dp[i], dp[j] + 1)
        
        # Return -1 if it's impossible to partition the string
        return dp[n] if dp[n] != float('inf') else -1