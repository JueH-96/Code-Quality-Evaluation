from collections import defaultdict

class Solution:
    def maximumLength(self, s: str) -> int:
        runs = self.get_runs(s)
        counts = defaultdict(int)
        
        for char, length in runs:
            for k in range(1, length + 1):
                contribution = length - k + 1
                counts[(char, k)] += contribution
        
        max_k = -1
        for (char, k), cnt in counts.items():
            if cnt >= 3 and k > max_k:
                max_k = k
        
        return max_k if max_k != -1 else -1
    
    def get_runs(self, s):
        if not s:
            return []
        runs = []
        current_char = s[0]
        count = 1
        for c in s[1:]:
            if c == current_char:
                count += 1
            else:
                runs.append((current_char, count))
                current_char = c
                count = 1
        runs.append((current_char, count))
        return runs