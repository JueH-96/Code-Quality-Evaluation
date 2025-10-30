from functools import lru_cache

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        
        @lru_cache(None)
        def dp(i, first, last):
            if i == n:
                return 0
            
            w = words[i]
            option1 = len(w) - (last == w[0])
            option2 = len(w) - (first == w[-1])
            
            return min(
                option1 + dp(i + 1, w[0], last),
                option2 + dp(i + 1, first, w[-1])
            )
        
        return len(words[0]) + dp(1, words[0][0], words[0][-1])