class Solution:
    def possibleStringCount(self, word: str) -> int:
        n = len(word)
        count = 1
        i = 0
        while i < n:
            j = i
            while j < n and word[j] == word[i]:
                j += 1
            length = j - i
            if length > 1:
                count *= length
            i = j
        return count