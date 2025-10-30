from typing import List

class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        count = 0
        n = len(words)
        for j in range(n):
            for i in range(j):
                if len(words[i]) > len(words[j]):
                    continue
                if words[j].startswith(words[i]) and words[j].endswith(words[i]):
                    count += 1
        return count