class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n, m = len(word1), len(word2)
        i, j, change_made = 0, 0, False
        result = []
        
        while i < n and j < m:
            if word1[i] == word2[j]:
                result.append(i)
                j += 1
            elif not change_made and word1[i] != word2[j]:
                if any(word1[i] == word2[k] for k in range(j, m)):
                    change_made = True
                    result.append(i)
                else:
                    i += 1
            else:
                i += 1
        
        return result if j == m else []