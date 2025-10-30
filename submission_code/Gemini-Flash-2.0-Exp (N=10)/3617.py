class Solution:
    def possibleStringCount(self, word: str) -> int:
        n = len(word)
        if n == 0:
            return 0
        
        count = 1
        
        i = 0
        while i < n:
            j = i
            while j < n and word[i] == word[j]:
                j += 1
            
            if j - i > 1:
                count += (j - i)
            
            i = j
        
        return count