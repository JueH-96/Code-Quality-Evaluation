from typing import List
from collections import Counter

class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        # Count the frequency of each element
        freq = Counter(nums)
        # Find the maximum frequency
        max_freq = max(freq.values())
        # Return the total number of elements that have the maximum frequency
        return sum(count for count in freq.values() if count == max_freq)