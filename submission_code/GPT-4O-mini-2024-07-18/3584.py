from typing import List

class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n, m = len(word1), len(word2)
        if m > n:
            return []

        indices = []
        j = 0
        
        for i in range(n):
            if j < m and word1[i] == word2[j]:
                indices.append(i)
                j += 1
            elif j < m and word1[i] != word2[j]:
                # Check if we can change this character to match word2[j]
                if j + 1 == m or word1[i] == word2[j + 1]:
                    indices.append(i)
                    j += 1
                if j == m:
                    break
        
        if len(indices) != m:
            return []
        
        # Check if the sequence is almost equal
        changes = 0
        for k in range(m):
            if word1[indices[k]] != word2[k]:
                changes += 1
        
        if changes > 1:
            return []
        
        return indices