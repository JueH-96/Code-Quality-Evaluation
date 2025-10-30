class Solution:
    def possibleStringCount(self, word: str) -> int:
        total = 1
        i, n = 0, len(word)
        while i < n:
            j = i
            while j < n and word[j] == word[i]:
                j += 1
            run_length = j - i
            if run_length > 1:
                total += (run_length - 1)
            i = j
        return total