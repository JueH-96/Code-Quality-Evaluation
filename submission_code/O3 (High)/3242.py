from typing import List
from collections import Counter

class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        """
        Computes the total number of occurrences (sum of frequencies) 
        of the elements that appear with the maximum frequency in `nums`.
        """
        # Count frequencies of each number
        freq = Counter(nums)

        # Find the maximum frequency among all elements
        max_freq = max(freq.values())

        # Sum the frequencies of all elements whose frequency equals max_freq
        total = sum(count for count in freq.values() if count == max_freq)
        
        return total