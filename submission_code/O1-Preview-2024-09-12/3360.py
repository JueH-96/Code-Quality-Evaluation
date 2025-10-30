class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from collections import Counter
        
        freq = Counter(word)
        counts = list(freq.values())
        unique_counts = sorted(set(counts), reverse=True)
        
        min_deletions = float('inf')
        
        for f_max in unique_counts:
            f_min = max(f_max - k, 0)
            deletions = 0
            for c in freq:
                freq_c = freq[c]
                if freq_c > f_max:
                    deletions += freq_c - f_max
                elif freq_c < f_min:
                    deletions += freq_c
                # else: deletions += 0
            min_deletions = min(min_deletions, deletions)
        
        return min_deletions