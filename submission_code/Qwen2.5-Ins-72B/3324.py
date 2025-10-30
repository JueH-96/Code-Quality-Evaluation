from typing import List
from collections import Counter

class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        # Count the frequency of each element in the array
        freq = Counter(nums)
        
        # Check if any element appears more than twice
        for count in freq.values():
            if count > 2:
                return False
        
        # If no element appears more than twice, it is possible to split the array
        return True