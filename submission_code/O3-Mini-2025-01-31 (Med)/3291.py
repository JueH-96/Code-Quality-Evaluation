from typing import List

class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        # Helper function to count the number of set bits in an integer.
        def popcount(x: int) -> int:
            return bin(x).count("1")
        
        # Calculate the sequence of set bit counts in the original array.
        original_groups = [popcount(x) for x in nums]
        # Calculate the sequence of set bit counts in the sorted array.
        sorted_nums = sorted(nums)
        sorted_groups = [popcount(x) for x in sorted_nums]
        
        # The relative order of group identities (popcount values) is invariant
        # because we can only swap adjacent elements that share the same popcount.
        # Thus, the operation can only reorder within each group. 
        # If the sorted array produces a different sequence of popcounts,
        # then it is impossible to reach that order.
        return original_groups == sorted_groups