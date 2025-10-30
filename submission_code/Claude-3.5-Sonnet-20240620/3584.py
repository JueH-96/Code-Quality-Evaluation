class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n, m = len(word1), len(word2)
        result = []
        
        def backtrack(i, j, changes, current):
            if j == m:
                nonlocal result
                if not result or current < result:
                    result = current[:]
                return
            
            if i == n:
                return
            
            if word1[i] == word2[j]:
                current.append(i)
                backtrack(i + 1, j + 1, changes, current)
                current.pop()
            
            if changes == 0:
                current.append(i)
                backtrack(i + 1, j + 1, 1, current)
                current.pop()
            
            backtrack(i + 1, j, changes, current)
        
        backtrack(0, 0, 0, [])
        return result