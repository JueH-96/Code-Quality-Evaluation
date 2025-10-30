from typing import List

class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        pairs = 0
        seen = set()
        
        for word in words:
            reversed_word = word[::-1]
            if reversed_word in seen:
                pairs += 1
                seen.remove(reversed_word)
            else:
                seen.add(word)
        
        return pairs