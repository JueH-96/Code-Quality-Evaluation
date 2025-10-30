from collections import Counter

class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        freqs = list(Counter(word).values())
        max_freq = max(freqs) if freqs else 0
        min_del = float('inf')
        
        for a in range(1, max_freq + 1):
            total = 0
            for f in freqs:
                if f < a:
                    total += f
                elif f > a + k:
                    total += f - (a + k)
            min_del = min(min_del, total)
        
        return min_del