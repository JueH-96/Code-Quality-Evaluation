from typing import List

class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n, m = len(word1), len(word2)
        diff = 0
        ans = []
        i, j = 0, 0
        
        while i < n and j < m:
            if word1[i] == word2[j]:
                ans.append(i)
                i += 1
                j += 1
            else:
                if diff == 0:
                    diff += 1
                    ans.append(i)
                    i += 1
                else:
                    if j == m - 1:
                        ans.append(i)
                        i += 1
                    else:
                        return []
        
        if j < m:
            return []
        
        return ans