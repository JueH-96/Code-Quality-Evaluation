class Solution:
    def possibleStringCount(self, word: str) -> int:
        if not word:
            return 0
        runs = []
        current_char = word[0]
        count = 1
        for i in range(1, len(word)):
            if word[i] == current_char:
                count += 1
            else:
                runs.append(count)
                current_char = word[i]
                count = 1
        runs.append(count)
        sum_total = sum((run - 1) for run in runs)
        return sum_total + 1