from collections import Counter
from typing import List

class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        counts = Counter(nums)
        if not counts:
            return 0

        max_freq = 0
        for count in counts.values():
            max_freq = max(max_freq, count)

        total_freq = 0
        for num, freq in counts.items():
            if freq == max_freq:
                total_freq += freq

        return total_freq