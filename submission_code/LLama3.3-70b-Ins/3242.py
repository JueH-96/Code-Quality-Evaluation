from typing import List
from collections import Counter

class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        # Count the frequency of each element in the array
        freq = Counter(nums)
        
        # Find the maximum frequency
        max_freq = max(freq.values())
        
        # Count the number of elements with the maximum frequency
        count = sum(val == max_freq for val in freq.values())
        
        # Return the total frequencies of elements with the maximum frequency
        return count * max_freq