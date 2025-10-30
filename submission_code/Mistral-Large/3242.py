from typing import List
from collections import Counter

class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        # Count the frequency of each element in the array
        frequency = Counter(nums)

        # Find the maximum frequency
        max_freq = max(frequency.values())

        # Calculate the total number of elements with the maximum frequency
        total_max_freq_elements = sum(count for elem, count in frequency.items() if count == max_freq)

        return total_max_freq_elements