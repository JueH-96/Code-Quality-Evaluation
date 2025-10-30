from collections import Counter

class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        freq = list(Counter(word).values())
        if not freq:
            return 0
        max_freq = max(freq)
        min_deletions = float('inf')
        
        for target_max in range(0, max_freq + 1):
            min_allowed = max(target_max - k, 0)
            deletions = 0
            for f in freq:
                if f > target_max:
                    deletions += f - target_max
                elif f < min_allowed:
                    deletions += f
            if deletions < min_deletions:
                min_deletions = deletions
        
        return min_deletions