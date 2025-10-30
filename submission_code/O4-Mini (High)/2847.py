from typing import List

class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        seen = set()
        count = 0
        for w in words:
            rw = w[::-1]
            if rw in seen:
                count += 1
                seen.remove(rw)
            else:
                seen.add(w)
        return count