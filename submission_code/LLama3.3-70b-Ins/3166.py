from typing import List
from collections import Counter

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        # Count the frequency of each number in the array
        freq = Counter(nums)
        
        # Initialize the minimum number of groups
        min_groups = 0
        
        # For each frequency, calculate the number of groups needed
        for count in freq.values():
            # The number of groups needed for this frequency is the ceiling of the frequency divided by the minimum number of elements per group
            # Since the difference between the number of indices assigned to two groups should not exceed 1, the minimum number of elements per group is the average of the total number of elements and the number of groups
            # We use the ceiling function to round up to the nearest integer
            min_groups = max(min_groups, (count + 1) // 2)
        
        # Return the minimum number of groups
        return len(freq) if min_groups == 1 else min_groups