class Solution:
    def possibleStringCount(self, word: str) -> int:
        runs = []
        current_char = word[0]
        count = 1
        for c in word[1:]:
            if c == current_char:
                count += 1
            else:
                runs.append(count)
                current_char = c
                count = 1
        runs.append(count)
        return sum(r - 1 for r in runs) + 1