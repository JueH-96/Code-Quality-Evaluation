from collections import Counter
import bisect

class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        counts = Counter(word)
        sorted_freq = sorted(counts.values())
        n = len(sorted_freq)
        if n == 0:
            return 0
        
        # Compute prefix sums
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + sorted_freq[i]
        
        max_freq = sorted_freq[-1]
        max_total = 0
        original_length = sum(sorted_freq)
        
        # Iterate all possible m from 0 to max_freq inclusive
        for m in range(max_freq + 1):
            lower_bound = m - k
            if lower_bound < 0:
                lower_bound = 0
            
            # Find the first index where frequency >= lower_bound
            idx = bisect.bisect_left(sorted_freq, lower_bound)
            subset = sorted_freq[idx:]
            
            if not subset:
                current_sum = 0
            else:
                # Find the position where elements exceed m
                pos = bisect.bisect_right(subset, m)
                sum_less = prefix_sum[idx + pos] - prefix_sum[idx]
                sum_more = m * (len(subset) - pos)
                current_sum = sum_less + sum_more
            
            if current_sum > max_total:
                max_total = current_sum
        
        return original_length - max_total