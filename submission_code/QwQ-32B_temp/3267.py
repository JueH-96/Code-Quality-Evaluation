from collections import defaultdict

class Solution:
    def maximumLength(self, s: str) -> int:
        runs = []
        if not s:
            return -1
        current_char = s[0]
        current_length = 1
        for c in s[1:]:
            if c == current_char:
                current_length += 1
            else:
                runs.append((current_char, current_length))
                current_char = c
                current_length = 1
        runs.append((current_char, current_length))
        
        char_runs = defaultdict(list)
        for char, length in runs:
            char_runs[char].append(length)
        
        max_len = -1
        
        for c in char_runs:
            runs_list = char_runs[c]
            max_possible = max(runs_list) if runs_list else 0
            for l in range(1, max_possible + 1):
                total = 0
                for run in runs_list:
                    if run >= l:
                        total += (run - l + 1)
                if total >= 3 and l > max_len:
                    max_len = l
        
        return max_len