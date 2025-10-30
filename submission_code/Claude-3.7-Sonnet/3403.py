class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        memo = {}
        
        def dp(start):
            if start == len(s):
                return 0
            
            if start in memo:
                return memo[start]
            
            min_partitions = float('inf')
            freq = {}
            
            for end in range(start, len(s)):
                # Count frequency of current character
                if s[end] in freq:
                    freq[s[end]] += 1
                else:
                    freq[s[end]] = 1
                
                # Check if substring s[start:end+1] is balanced
                # (all characters appear the same number of times)
                values = list(freq.values())
                if len(set(values)) == 1:
                    # If balanced, try this partition and solve for the rest
                    partitions = 1 + dp(end + 1)
                    min_partitions = min(min_partitions, partitions)
            
            memo[start] = min_partitions
            return min_partitions
        
        return dp(0)