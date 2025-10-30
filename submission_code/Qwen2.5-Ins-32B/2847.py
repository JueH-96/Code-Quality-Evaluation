from typing import List

class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        pairs = 0
        seen = set()
        for word in words:
            if word[::-1] in seen:
                pairs += 1
                seen.remove(word[::-1])
            else:
                seen.add(word)
        return pairs