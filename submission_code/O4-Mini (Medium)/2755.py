from typing import List

class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        # dp[i] = minimum extra characters in s[i:]
        dp = [float('inf')] * (n + 1)
        dp[n] = 0
        
        dict_set = set(dictionary)
        max_len = max((len(w) for w in dictionary), default=0)
        
        for i in range(n - 1, -1, -1):
            # Option 1: treat s[i] as an extra character
            dp[i] = dp[i + 1] + 1
            
            # Option 2: try to match any dictionary word starting at i
            # We only need to check lengths up to max_len
            for l in range(1, max_len + 1):
                if i + l <= n and s[i:i+l] in dict_set:
                    dp[i] = min(dp[i], dp[i + l])
        
        return dp[0]