class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # base case: empty string requires 0 partitions
        
        for i in range(1, n + 1):
            freqs = {}
            for j in range(i - 1, -1, -1):
                char = s[j]
                freqs[char] = freqs.get(char, 0) + 1
                if self.is_balanced(freqs):
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n]
    
    def is_balanced(self, freqs):
        counts = list(freqs.values())
        # Filter out characters that don't appear in the substring
        counts = [c for c in counts if c > 0]
        if len(set(counts)) == 1:
            return True
        return False