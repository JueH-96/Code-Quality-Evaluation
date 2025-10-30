class Solution:
    def possibleStringCount(self, word: str) -> int:
        result = 1
        i = 0
        while i < len(word):
            char = word[i]
            length = 1
            j = i + 1
            while j < len(word) and word[j] == char:
                length += 1
                j += 1
            if length >= 2:
                result += (length - 1)
            i = j
        return result