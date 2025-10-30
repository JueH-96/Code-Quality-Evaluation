class Solution:
    def possibleStringCount(self, word: str) -> int:
        n = len(word)
        if n == 1:
            return 1
        count = 0
        i = 0
        while i < n:
            current_char = word[i]
            j = i
            while j < n and word[j] == current_char:
                j += 1
            length = j - i
            if length > 1:
                count += 1
            i = j
        if count == 0:
            return 1
        else:
            return count + 1