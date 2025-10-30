class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Generate all powers of 5 that fit within 15 bits
        powers_of_5 = set()
        power = 1
        while power.bit_length() <= 15:
            powers_of_5.add(bin(power)[2:])  # Convert to binary string without '0b'
            power *= 5
        
        n = len(s)
        # dp[i] represents the minimum number of beautiful substrings to partition s[0:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Empty string needs 0 partitions
        
        for i in range(1, n + 1):
            # Try all possible starting positions for the last substring
            for j in range(i):
                substring = s[j:i]
                # Check if substring is beautiful
                if substring[0] != '0' and substring in powers_of_5:
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n] if dp[n] != float('inf') else -1