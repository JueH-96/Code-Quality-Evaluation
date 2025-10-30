class Solution:
    def possibleStringCount(self, word: str) -> int:
        total = 1
        n = len(word)
        if n == 0:
            return total
        current_char = word[0]
        run_length = 1
        for i in range(1, n):
            if word[i] == current_char:
                run_length += 1
            else:
                if run_length >= 2:
                    total += run_length - 1
                current_char = word[i]
                run_length = 1
        if run_length >= 2:
            total += run_length - 1
        return total