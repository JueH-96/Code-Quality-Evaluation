from typing import List

class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        # Convert words into a set for O(1) lookups.
        words_set = set(words)
        pairs = 0
        for word in words:
            rev = word[::-1]
            # To ensure we count each pair only once and avoid self pairing (e.g. "zz"),
            # we only count the pair when word is lexicographically smaller than its reverse.
            if rev in words_set and word < rev:
                pairs += 1
        return pairs