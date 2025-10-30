from typing import List
from collections import Counter

class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        # Count the frequency of each element in the list
        frequency = Counter(nums)
        # Find the maximum frequency among all elements
        max_freq = max(frequency.values())
        # Sum the counts (total occurrences) for elements with the maximum frequency
        total_freq = sum(count for count in frequency.values() if count == max_freq)
        return total_freq