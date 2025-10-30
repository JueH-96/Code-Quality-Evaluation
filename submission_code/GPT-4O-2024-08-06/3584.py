from typing import List

class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n, m = len(word1), len(word2)
        if m >= n:
            return []

        # To store the result indices
        result = []
        # To track the current position in word2
        j = 0
        # To track the number of mismatches
        mismatch_count = 0

        for i in range(n):
            if j < m and word1[i] == word2[j]:
                result.append(i)
                j += 1
            elif j < m and mismatch_count == 0:
                # Allow one mismatch
                result.append(i)
                mismatch_count += 1
                j += 1

            if j == m:
                break

        # Check if we have matched all characters of word2
        if j == m:
            return result
        else:
            return []