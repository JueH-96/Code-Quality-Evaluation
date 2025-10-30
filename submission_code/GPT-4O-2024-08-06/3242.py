from typing import List
from collections import Counter

class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        # Count the frequency of each element in the list
        frequency = Counter(nums)
        
        # Find the maximum frequency
        max_freq = max(frequency.values())
        
        # Sum the frequencies of elements that have the maximum frequency
        total_max_freq_elements = sum(freq for freq in frequency.values() if freq == max_freq)
        
        return total_max_freq_elements