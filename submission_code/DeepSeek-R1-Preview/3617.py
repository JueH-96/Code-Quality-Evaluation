class Solution:
    def possibleStringCount(self, word: str) -> int:
        if not word:
            return 0
        
        runs = []
        prev_char = word[0]
        count = 1
        
        for c in word[1:]:
            if c == prev_char:
                count += 1
            else:
                runs.append(count)
                prev_char = c
                count = 1
        runs.append(count)  # Add the last run
        
        total = sum((run - 1) for run in runs) + 1
        return total