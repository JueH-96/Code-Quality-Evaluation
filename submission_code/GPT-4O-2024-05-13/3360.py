from collections import Counter

class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        freq = Counter(word)
        freq_values = list(freq.values())
        freq_values.sort()
        
        min_deletions = float('inf')
        
        for i in range(len(freq_values)):
            max_freq = freq_values[i]
            deletions = 0
            
            for f in freq_values:
                if f > max_freq + k:
                    deletions += f - (max_freq + k)
                elif f < max_freq - k:
                    deletions += f
            
            min_deletions = min(min_deletions, deletions)
        
        return min_deletions