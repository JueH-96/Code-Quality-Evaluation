class Solution:
    def possibleStringCount(self, word: str) -> int:
        count = 1
        i = 0
        while i < len(word):
            char = word[i]
            j = i + 1
            while j < len(word) and word[j] == char:
                j += 1
            length = j - i
            if length > 1:
                count += (length - 1)
            i = j
        return count