from collections import Counter
from typing import List

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        # Count the frequency of each number in the array
        freq = Counter(nums)
        
        # The minimum number of groups needed will be the maximum frequency of any number
        return max(freq.values())