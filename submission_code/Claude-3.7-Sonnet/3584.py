from typing import List

class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n, m = len(word1), len(word2)
        
        def dfs(word2_idx, start_index, mismatches, current_indices):
            # If we've matched all characters in word2
            if word2_idx == m:
                return current_indices.copy()
            
            # If we have more than one mismatch, prune this branch
            if mismatches > 1:
                return None
            
            # Try indices from start_index to len(word1) - 1
            for i in range(start_index, n):
                current_indices.append(i)
                if word1[i] == word2[word2_idx]:
                    result = dfs(word2_idx + 1, i + 1, mismatches, current_indices)
                else:
                    result = dfs(word2_idx + 1, i + 1, mismatches + 1, current_indices)
                
                if result:
                    return result
                
                current_indices.pop()
            
            return None
        
        return dfs(0, 0, 0, []) or []