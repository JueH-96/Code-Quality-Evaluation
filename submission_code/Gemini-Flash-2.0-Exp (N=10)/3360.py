from collections import Counter

class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        counts = Counter(word)
        freqs = sorted(counts.values())
        n = len(freqs)
        min_deletions = len(word)

        for i in range(n):
            for j in range(i, n):
                min_freq = freqs[i]
                max_freq = freqs[j]
                if max_freq - min_freq <= k:
                    deletions = 0
                    for freq in freqs:
                        if freq < min_freq or freq > max_freq:
                            deletions += freq
                    min_deletions = min(min_deletions, deletions)
        
        return min_deletions