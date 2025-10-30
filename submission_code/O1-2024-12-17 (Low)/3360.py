class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from collections import Counter

        freq_counter = Counter(word)
        freqs = list(freq_counter.values())
        max_freq = max(freqs)

        # We'll try all possible "max frequency" values from 0 up to max_freq.
        # For each, the implied min frequency is (max_freq_value - k) if positive, else 0.
        # We compute how many deletions are needed to force each character's frequency
        # into the interval [L, R], where R = candidate max frequency, L = max(0, R - k).
        best = sum(freqs)  # Worst case: remove everything
        for R in range(max_freq + 1):
            L = max(0, R - k)
            deletions = 0
            for f in freqs:
                if f > R:
                    deletions += (f - R)  # reduce down to R
                elif f < L:
                    deletions += f       # remove entirely
            best = min(best, deletions)

        return best