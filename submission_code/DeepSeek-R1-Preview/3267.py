from collections import defaultdict

class Solution:
    def maximumLength(self, s: str) -> int:
        # Process the string into runs of characters
        runs = defaultdict(list)
        current_char = s[0]
        count = 1
        for i in range(1, len(s)):
            if s[i] == current_char:
                count += 1
            else:
                runs[current_char].append(count)
                current_char = s[i]
                count = 1
        # Add the last run
        runs[current_char].append(count)
        
        max_k = -1
        
        # Check each character's runs
        for c in runs:
            runs_c = runs[c]
            max_run_c = max(runs_c)
            # Check possible k's from max_run_c down to 1
            for k in range(max_run_c, 0, -1):
                total = 0
                for run in runs_c:
                    if run >= k:
                        total += (run - k + 1)
                if total >= 3:
                    if k > max_k:
                        max_k = k
                    break  # No need to check smaller k for this character
        
        return max_k if max_k != -1 else -1