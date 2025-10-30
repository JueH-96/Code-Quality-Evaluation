from typing import List

class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        count = 0
        n = len(words)
        for j in range(n):
            for i in range(j):
                str1 = words[i]
                str2 = words[j]
                if len(str1) > len(str2):
                    continue
                if str2.startswith(str1) and str2.endswith(str1):
                    count += 1
        return count