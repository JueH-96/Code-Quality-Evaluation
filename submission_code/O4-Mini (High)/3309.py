from typing import List

class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        count = 0
        n = len(words)
        for i in range(n):
            for j in range(i + 1, n):
                w1 = words[i]
                w2 = words[j]
                # Check if w1 is both a prefix and a suffix of w2
                if len(w1) <= len(w2) and w2.startswith(w1) and w2.endswith(w1):
                    count += 1
        return count