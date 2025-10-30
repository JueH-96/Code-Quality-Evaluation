from typing import List
from collections import defaultdict

class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        # Helper function to compute number of set bits
        def pop_count(x: int) -> int:
            return bin(x).count("1")
        
        # Group the numbers based on their pop count in the order they appear.
        groups = defaultdict(list)
        for num in nums:
            groups[pop_count(num)].append(num)
        
        # Within each pop count group, we can sort arbitrarily.
        # So, sort each group separately.
        for k in groups:
            groups[k].sort()
        
        # Now, reconstruct what sorted array we can achieve.
        # For each original element, we choose the smallest available number 
        # in its pop count group, because we can only rearrange within the group.
        result = []
        for num in nums:
            p = pop_count(num)
            # take the next smallest number from group p
            result.append(groups[p].pop(0))
        
        # Check if the reconstructed list is sorted in non-decreasing order.
        return all(result[i] <= result[i+1] for i in range(len(result) - 1))