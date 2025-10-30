class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Generate all powers of 5 in binary up to length 15
        powers_of_5 = []
        power = 1
        while True:
            binary = bin(power)[2:]  # Remove '0b' prefix
            if len(binary) > 15:
                break
            powers_of_5.append(binary)
            power *= 5
        
        # Convert to set for O(1) lookup
        powers_set = set(powers_of_5)
        
        n = len(s)
        # dp[i] = minimum number of beautiful substrings to partition s[0:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            for j in range(i):
                if s[j] == '0':  # Skip substrings with leading zeros
                    continue
                substring = s[j:i]
                # Check if substring is a power of 5
                if substring in powers_set:
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n] if dp[n] != float('inf') else -1