from typing import List

class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        count = 0
        n = len(words)
        for i in range(n):
            s1 = words[i]
            for j in range(i+1, n):
                s2 = words[j]
                # Only consider if s1 can be both prefix and suffix of s2
                if len(s1) <= len(s2) and s2.startswith(s1) and s2.endswith(s1):
                    count += 1
        return count