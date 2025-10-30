class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        from collections import Counter
        
        def is_balanced(counter):
            values = list(counter.values())
            return all(x == values[0] for x in values)
        
        n = len(s)
        if n == 0:
            return 0
        
        # This will store the minimum number of partitions needed for the string up to index i
        dp = [float('inf')] * n
        dp[0] = 1  # Base case: a single character is always balanced
        
        for i in range(n):
            char_count = Counter()
            # Check every possible substring ending at i
            for j in range(i, -1, -1):
                char_count[s[j]] += 1
                if is_balanced(char_count):
                    if j == 0:
                        dp[i] = 1  # The whole substring from 0 to i is balanced
                    else:
                        dp[i] = min(dp[i], dp[j-1] + 1)
        
        return dp[-1]