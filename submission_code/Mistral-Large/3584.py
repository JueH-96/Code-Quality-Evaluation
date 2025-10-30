from typing import List

class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        m, n = len(word1), len(word2)
        j = 0
        diff = 0
        result = []

        for i in range(m):
            if j < n and word1[i] == word2[j]:
                result.append(i)
                j += 1
            elif diff == 0 and j < n:
                result.append(i)
                diff = 1
            elif j < n and diff == 1:
                j += 1
                diff = 2

        if j == n and (diff == 1 or diff == 0):
            return result
        else:
            return []