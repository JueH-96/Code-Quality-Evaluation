from collections import Counter

class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        freq = Counter(word)
        freq_values = sorted(freq.values())
        
        min_deletions = float('inf')
        
        for i in range(len(freq_values)):
            target = freq_values[i]
            deletions = 0
            for j in range(len(freq_values)):
                if freq_values[j] < target:
                    deletions += freq_values[j]
                elif freq_values[j] > target + k:
                    deletions += freq_values[j] - (target + k)
            min_deletions = min(min_deletions, deletions)
        
        return min_deletions