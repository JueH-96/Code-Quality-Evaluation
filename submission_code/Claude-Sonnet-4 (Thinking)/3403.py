class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        
        def is_balanced(substr):
            count = {}
            for char in substr:
                count[char] = count.get(char, 0) + 1
            
            frequencies = set(count.values())
            return len(frequencies) == 1
        
        # dp[i] represents minimum partitions needed for s[0:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # empty string needs 0 partitions
        
        for i in range(1, n + 1):
            for j in range(i):
                substr = s[j:i]
                if is_balanced(substr):
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n]