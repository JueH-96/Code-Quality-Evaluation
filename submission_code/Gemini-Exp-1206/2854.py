from typing import List
from functools import lru_cache

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        @lru_cache(maxsize=None)
        def solve(i, first, last):
            if i == len(words):
                return 0
            
            word = words[i]
            res1 = len(word)
            if last == word[0]:
                res1 -= 1
            res1 += solve(i + 1, first, word[-1])

            res2 = len(word)
            if word[-1] == first:
                res2 -= 1
            res2 += solve(i + 1, word[0], last)

            return min(res1, res2)

        return len(words[0]) + solve(1, words[0][0], words[0][-1])