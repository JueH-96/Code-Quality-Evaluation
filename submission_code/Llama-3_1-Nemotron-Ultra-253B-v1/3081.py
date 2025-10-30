from collections import Counter
from typing import List

class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        freq = Counter(nums)
        sorted_values = sorted(freq.keys(), reverse=True)
        available = 0
        total_pairs = 0
        for val in sorted_values:
            count = freq[val]
            pairs = min(count, available)
            total_pairs += pairs
            available = (count - pairs) + (available - pairs)
        return len(nums) - 2 * total_pairs