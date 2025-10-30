class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Precompute valid binary strings that represent a power of 5.
        valid = set()
        x = 1
        max_len = len(s)
        while len(bin(x)[2:]) <= max_len:
            valid.add(bin(x)[2:])
            x *= 5
            
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        # Use dynamic programming where dp[i] represents the minimum beautiful substrings for s[:i]
        for i in range(1, n + 1):
            for j in range(i):
                # Skip substrings that start with '0' as they have a leading zero.
                if s[j] == '0':
                    continue
                if s[j:i] in valid:
                    dp[i] = min(dp[i], dp[j] + 1)
                    
        return dp[n] if dp[n] != float('inf') else -1