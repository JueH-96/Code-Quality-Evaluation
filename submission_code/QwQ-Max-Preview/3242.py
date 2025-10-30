from collections import Counter
from typing import List

class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        counts = Counter(nums)
        max_count = max(counts.values()) if counts else 0
        return sum(count for count in counts.values() if count == max_count)