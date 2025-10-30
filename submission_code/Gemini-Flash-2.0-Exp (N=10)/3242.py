from typing import List
from collections import Counter

class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        counts = Counter(nums)
        max_freq = 0
        for count in counts.values():
            max_freq = max(max_freq, count)
        
        result = 0
        for num in nums:
            if counts[num] == max_freq:
                result += 1
        return result