from typing import List

class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        count = 0
        n = len(words)
        for i in range(n):
            for j in range(i + 1, n):
                prefix = words[i]
                candidate = words[j]
                # For prefix to be valid, its length should not exceed candidate's length.
                if len(prefix) <= len(candidate):
                    if candidate.startswith(prefix) and candidate.endswith(prefix):
                        count += 1
        return count