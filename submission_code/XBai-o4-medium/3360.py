class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from collections import Counter
        freq = list(Counter(word).values())
        if not freq:
            return 0
        max_freq = max(freq)
        min_deletions = float('inf')
        
        for x in range(0, max_freq + 1):
            total = 0
            for f in freq:
                if f < x:
                    total += f
                else:
                    upper = x + k
                    target = min(f, upper)
                    total += (f - target)
            if total < min_deletions:
                min_deletions = total
        
        return min_deletions