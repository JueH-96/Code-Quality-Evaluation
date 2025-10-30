from typing import List

class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        """
        Constructs the transformed array as described in the problem.
        For each element at index i, we take nums[i] steps (positive → right, negative → left)
        on the circular array and place the value found at that landing index into result[i].
        """
        
        n = len(nums)
        result = [0]*n
        
        for i, step in enumerate(nums):
            if step == 0:
                # When nums[i] is zero, result[i] is itself zero (nums[i]).
                result[i] = nums[i]
            else:
                # Calculate landing position in a circular manner.
                landing_idx = (i + step) % n
                result[i] = nums[landing_idx]
        
        return result