from collections import Counter
from typing import List

class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        counts = Counter(nums)
        max_freq = max(counts.values())
        num_elements_with_max = sum(1 for freq in counts.values() if freq == max_freq)
        return max_freq * num_elements_with_max