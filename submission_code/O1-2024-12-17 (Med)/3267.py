class Solution:
    def maximumLength(self, s: str) -> int:
        n = len(s)
        best = -1
        
        # Check each character from 'a' to 'z'
        for c in 'abcdefghijklmnopqrstuvwxyz':
            # Find runs of the character c in s
            runs = []
            start = None
            for i, ch in enumerate(s):
                if ch == c:
                    if start is None:
                        start = i
                else:
                    if start is not None:
                        runs.append(i - start)
                        start = None
            if start is not None:
                runs.append(n - start)
            
            # Check from longest possible length down to 1
            for length in range(n, 0, -1):
                count_occurrences = 0
                # For each run of c, see how many times a substring of this length can appear
                for run_len in runs:
                    if run_len >= length:
                        count_occurrences += (run_len - length + 1)
                if count_occurrences >= 3:
                    best = max(best, length)
                    break
        
        return best