class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        
        # dp[i] = minimum number of balanced substrings to partition s[0:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        def is_balanced(start, end):
            # Check if s[start:end] is balanced
            freq = {}
            for i in range(start, end):
                freq[s[i]] = freq.get(s[i], 0) + 1
            
            # All characters must appear the same number of times
            counts = list(freq.values())
            return len(set(counts)) == 1
        
        # Try all possible partitions
        for i in range(1, n + 1):
            for j in range(i):
                if is_balanced(j, i):
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n]