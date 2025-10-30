from typing import List

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        from functools import lru_cache

        @lru_cache(None)
        def dp(i, left, right):
            if i == len(words):
                return 0
            
            word = words[i]
            # Option 1: join str_i-1 + words[i]
            option1 = dp(i + 1, left, word[-1]) + len(word)
            if right == word[0]:
                option1 -= 1
            
            # Option 2: join words[i] + str_i-1
            option2 = dp(i + 1, word[0], right) + len(word)
            if left == word[-1]:
                option2 -= 1
            
            return min(option1, option2)
        
        return len(words[0]) + dp(1, words[0][0], words[0][-1])