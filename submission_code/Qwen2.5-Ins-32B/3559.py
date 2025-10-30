from typing import List

class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        from collections import defaultdict
        
        # Create a dictionary to store the longest prefix of each character
        prefix_dict = defaultdict(int)
        for word in words:
            for i in range(len(word)):
                prefix_dict[word[:i+1]] = max(prefix_dict[word[:i+1]], i+1)
        
        # Dynamic programming to find the minimum number of valid strings
        dp = [float('inf')] * (len(target) + 1)
        dp[0] = 0
        
        for i in range(1, len(target) + 1):
            for j in range(i):
                if target[j:i] in prefix_dict:
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[-1] if dp[-1] != float('inf') else -1