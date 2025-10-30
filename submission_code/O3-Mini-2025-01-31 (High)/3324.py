from typing import List
from collections import Counter

class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        # Count the frequency of each number in nums.
        freq = Counter(nums)
        
        # For a valid split, no number can appear more than twice.
        # If any number appears more than twice, one of the parts would get a duplicate.
        for count in freq.values():
            if count > 2:
                return False
        
        # If we passed the check, then a valid split is always possible.
        return True