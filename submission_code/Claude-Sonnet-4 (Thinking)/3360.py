class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from collections import Counter
        
        freq = list(Counter(word).values())
        
        min_deletions = float('inf')
        
        # Try min_freq = 0 and min_freq = each frequency in the list
        candidates = [0] + freq
        for min_freq in candidates:
            max_freq = min_freq + k
            deletions = 0
            for f in freq:
                if f > max_freq:
                    deletions += f - max_freq
                elif f < min_freq:
                    deletions += f
            min_deletions = min(min_deletions, deletions)
        
        return min_deletions