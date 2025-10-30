from typing import List

class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        pairs = 0
        word_set = set(words)
        
        # To avoid double counting, we only count a pair (word, rev)
        # if word < rev (lexicographically); note that word != rev because words are distinct.
        for word in words:
            rev = word[::-1]
            if rev in word_set and word < rev:
                pairs += 1
        return pairs