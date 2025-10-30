from collections import Counter

class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        freq_counter = Counter(word)
        freqs = list(freq_counter.values())
        max_freq_val = max(freqs)
        max_remaining = 0
        for H in range(1, max_freq_val + 1):
            L_min = max(1, H - k)
            current_sum = 0
            for freq in freqs:
                if freq >= L_min:
                    current_sum += min(freq, H)
            if current_sum > max_remaining:
                max_remaining = current_sum
        total_length = len(word)
        min_deletions = total_length - max_remaining
        return min_deletions