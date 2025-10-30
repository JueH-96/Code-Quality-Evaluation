class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        def is_balanced(start, end):
            freq = {}
            for i in range(start, end):
                freq[s[i]] = freq.get(s[i], 0) + 1
            return len(set(freq.values())) == 1
        
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            for j in range(i):
                if is_balanced(j, i):
                    dp[i] = min(dp[i], dp[j] + 1)
                    
        return dp[n]