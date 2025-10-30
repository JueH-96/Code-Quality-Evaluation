from collections import Counter

class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        freqs = list(Counter(word).values())
        total = sum(freqs)
        max_freq = max(freqs)
        max_sum_kept = 0
        
        for x in range(1, max_freq + 1):
            y = x + k
            sum_kept = 0
            for f in freqs:
                if f < x:
                    continue
                elif f > y:
                    sum_kept += y
                else:
                    sum_kept += f
            if sum_kept > max_sum_kept:
                max_sum_kept = sum_kept
                
        return total - max_sum_kept