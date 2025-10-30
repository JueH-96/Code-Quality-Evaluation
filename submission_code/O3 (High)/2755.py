from typing import List

class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        word_set = set(dictionary)                 # O(50) : quick membership test
        max_len = max(len(w) for w in dictionary)  # longest word length helps pruning
        
        # dp[i] = minimum extra characters in s[i:]
        dp = [0]*(n+1)
        dp[n] = 0                                  # empty suffix has 0 extra
        
        # iterate from right to left
        for i in range(n-1, -1, -1):
            # Option 1: treat s[i] as an extra character
            best = 1 + dp[i+1]
            
            # Option 2: match any dictionary word that starts at i
            # We only need to check lengths up to max_len to stay in bounds
            for l in range(1, max_len+1):
                if i + l > n:
                    break
                if s[i:i+l] in word_set:           # dictionary word fits here
                    best = min(best, dp[i+l])
            
            dp[i] = best
        
        return dp[0]