from collections import Counter
import bisect

class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        # Compute frequencies of characters present in the word
        freq = list(Counter(word).values())
        if not freq:
            return 0
        freq.sort()
        n = len(freq)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + freq[i]
        max_freq = max(freq)
        original_sum = sum(freq)
        max_total = 0
        
        for L in range(0, max_freq + 1):
            target = L + k
            # Find first index >= L
            idx_low = bisect.bisect_left(freq, L)
            # Find first index > target
            idx_high = bisect.bisect_right(freq, target)
            sum_low = prefix[idx_high] - prefix[idx_low]
            count_high = n - idx_high
            sum_high = count_high * target
            total = sum_low + sum_high
            if total > max_total:
                max_total = total
        
        return original_sum - max_total