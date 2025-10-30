from typing import List
from collections import Counter

class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        # Count the frequency of each element in the array
        count = Counter(nums)

        # Check if it's possible to split the array into two parts with distinct elements
        for value in count.values():
            if value > (len(nums) // 2):
                return False

        return True