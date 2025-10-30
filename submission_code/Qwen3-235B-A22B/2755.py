from typing import List

class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        word_set = set(dictionary)
        n = len(s)
        dp = [0] + [float('inf')] * n  # dp[0] is 0, others initialized to infinity
        
        for i in range(1, n + 1):
            # Start with the previous character plus one extra
            dp[i] = dp[i - 1] + 1
            
            # Check all possible j < i to find valid words ending at i
            for j in range(i):
                if s[j:i] in word_set:
                    dp[i] = min(dp[i], dp[j])
        
        return dp[n]