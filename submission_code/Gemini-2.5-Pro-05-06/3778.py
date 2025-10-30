from typing import List

class Solution:
    def transformArray(self, nums: List[int]) -> List[int]:
        # Step 1 & 2: Replace even numbers with 0 and odd numbers with 1.
        # A list comprehension is used here to create a new list `modified_nums`.
        # For each number `num` in the input list `nums`:
        # - If `num` is even (`num % 2 == 0`), it's replaced by 0.
        # - If `num` is odd (`num % 2 != 0`), it's replaced by 1.
        # Example: if nums = [4,3,2,1], modified_nums becomes [0,1,0,1].
        modified_nums = [0 if num % 2 == 0 else 1 for num in nums]
        
        # Step 3: Sort the modified array in non-decreasing order.
        # The `sort()` method sorts the list `modified_nums` in-place.
        # Since `modified_nums` only contains 0s and 1s, all 0s will come before all 1s.
        # Example: [0,1,0,1] becomes [0,0,1,1].
        modified_nums.sort()
        
        # Return the resulting sorted array.
        return modified_nums