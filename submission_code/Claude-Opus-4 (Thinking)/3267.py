class Solution:
    def maximumLength(self, s: str) -> int:
        from collections import defaultdict
        
        # For each character, store the lengths of all maximal runs
        char_runs = defaultdict(list)
        
        i = 0
        n = len(s)
        while i < n:
            j = i
            while j < n and s[j] == s[i]:
                j += 1
            # s[i:j] is a maximal run of character s[i]
            char_runs[s[i]].append(j - i)
            i = j
        
        max_length = -1
        
        # For each character, find the maximum length that occurs at least 3 times
        for char, runs in char_runs.items():
            # Try all possible lengths
            max_run = max(runs) if runs else 0
            for length in range(1, max_run + 1):
                count = 0
                for run_length in runs:
                    if run_length >= length:
                        # This run can contribute (run_length - length + 1) occurrences
                        count += run_length - length + 1
                
                if count >= 3:
                    max_length = max(max_length, length)
        
        return max_length