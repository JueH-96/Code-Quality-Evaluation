class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Precompute all powers of 5 (in binary form) up to length 15
        powers = set()
        p = 1
        while True:
            b = bin(p)[2:]  # convert decimal to binary string (strip off '0b')
            if len(b) > 15:
                break
            powers.add(b)
            p *= 5
        
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[n] = 0  # base case: minimum cuts to partition empty substring is 0
        
        # Bottom-up DP: dp[i] = min number of beautiful substrings s[i:]
        for i in range(n - 1, -1, -1):
            # Only need to consider up to 15 bits (since largest power of 5 binary under length 15)
            for j in range(i + 1, min(n + 1, i + 16)):
                if s[i:j] in powers:
                    dp[i] = min(dp[i], 1 + dp[j])
        
        return dp[0] if dp[0] != float('inf') else -1