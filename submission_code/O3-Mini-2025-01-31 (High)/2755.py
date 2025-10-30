from typing import List
from functools import lru_cache

class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        wordSet = set(dictionary)
        n = len(s)
        
        @lru_cache(maxsize=None)
        def dp(i: int) -> int:
            # base case: no characters left
            if i == n:
                return 0
            
            # Option 1: treat current character as extra
            ans = 1 + dp(i + 1)
            
            # Option 2: try to match each possible dictionary word starting at i
            # Instead of iterating over dictionary words, we can try all j from i to n-1
            # and if s[i:j+1] is in dictionary, then no extra char for those positions.
            for j in range(i, n):
                if s[i:j+1] in wordSet:
                    ans = min(ans, dp(j + 1))
            return ans
        
        return dp(0)