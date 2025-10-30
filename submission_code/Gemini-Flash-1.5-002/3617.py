class Solution:
    def possibleStringCount(self, word: str) -> int:
        count = 1
        i = 0
        while i < len(word):
            j = i
            while j + 1 < len(word) and word[j+1] == word[j]:
                j += 1
            
            if j > i:
                count *= (j - i + 2)
            i = j + 1

        return count