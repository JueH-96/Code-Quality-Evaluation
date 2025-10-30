from typing import List

class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        seen = set()
        count = 0
        
        for w in words:
            rev = w[::-1]
            if rev in seen:
                # Found a matching reverse, form a pair and remove the used one.
                count += 1
                seen.remove(rev)
            else:
                # Otherwise, keep this word for potential future pairing.
                seen.add(w)
        
        return count