from collections import Counter

class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        freq = Counter(word)
        freqs = sorted(freq.values(), reverse=True)
        
        if len(freqs) == 1:
            return 0
        
        min_deletions = float('inf')
        
        for i in range(len(freqs) - 1):
            for j in range(i + 1, len(freqs)):
                if abs(freqs[i] - freqs[j]) <= k:
                    deletions = sum(freqs[:i]) + sum(freqs[j+1:])
                    min_deletions = min(min_deletions, deletions)
                    break
        
        return min(min_deletions, sum(freqs[:-1]))