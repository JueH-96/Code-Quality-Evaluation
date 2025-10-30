class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        
        # dp[i] = minimum number of balanced substrings to partition s[0:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            freq = {}
            for j in range(i - 1, -1, -1):
                # Add s[j] to the frequency map
                freq[s[j]] = freq.get(s[j], 0) + 1
                
                # Check if s[j:i] is balanced
                values = list(freq.values())
                if all(v == values[0] for v in values):
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n]