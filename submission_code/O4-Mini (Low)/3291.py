from typing import List

class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        # Compute popcount for a number
        def popcount(x: int) -> int:
            return bin(x).count('1')
        
        # Original sequence of popcounts
        original_pcs = [popcount(x) for x in nums]
        # Sequence of popcounts after sorting nums
        sorted_pcs = [popcount(x) for x in sorted(nums)]
        
        # We can only swap adjacent elements with the same popcount,
        # so the popcount sequence must remain unchanged.
        return original_pcs == sorted_pcs