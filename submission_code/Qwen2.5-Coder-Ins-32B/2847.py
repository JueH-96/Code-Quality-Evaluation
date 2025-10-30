from typing import List

class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        seen = set()
        count = 0
        for word in words:
            if word in seen:
                count += 1
            seen.add(word[::-1])
        return count