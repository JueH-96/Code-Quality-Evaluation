class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from collections import Counter
        
        counts = Counter(word)
        freqs = list(counts.values())
        if not freqs:
            return 0  # empty word, but constraints say word length >=1
        
        max_freq = max(freqs)
        min_deletions = float('inf')
        
        for T in range(0, max_freq + 1):
            current_cost = 0
            for f in freqs:
                lower_bound = T - k
                if f < lower_bound:
                    current_cost += f  # must delete all of them
                elif f > T:
                    current_cost += (f - T)
            if current_cost < min_deletions:
                min_deletions = current_cost
        
        return min_deletions