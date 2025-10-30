from typing import List

class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        seen = set()
        pairs = 0
        
        for word in words:
            rev = word[::-1]
            if rev in seen:
                pairs += 1
                seen.remove(rev)
            else:
                seen.add(word)
        
        return pairs