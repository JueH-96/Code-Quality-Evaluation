from collections import defaultdict

class Solution:
    def maximumLength(self, s: str) -> int:
        runs = defaultdict(list)
        current_char = s[0]
        current_run = 1
        for c in s[1:]:
            if c == current_char:
                current_run += 1
            else:
                runs[current_char].append(current_run)
                current_char = c
                current_run = 1
        runs[current_char].append(current_run)
        
        max_L = len(s) - 2
        for L in range(max_L, 0, -1):
            for c in runs:
                total = 0
                for run in runs[c]:
                    if run >= L:
                        total += run - L + 1
                if total >= 3:
                    return L
        return -1