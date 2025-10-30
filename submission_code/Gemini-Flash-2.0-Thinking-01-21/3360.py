import collections

class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        counts = collections.Counter(word)
        frequencies = list(counts.values())
        min_deletions = float('inf')
        max_original_freq = 0
        if frequencies:
            max_original_freq = max(frequencies)
        
        possible_max_frequencies = sorted(list(set([0] + frequencies)))

        for max_f in possible_max_frequencies:
            min_f = max(0, max_f - k)
            current_deletions = 0
            for freq in frequencies:
                if freq < min_f:
                    current_deletions += freq
                elif freq > max_f:
                    current_deletions += (freq - max_f)
            min_deletions = min(min_deletions, current_deletions)
            
        return min_deletions