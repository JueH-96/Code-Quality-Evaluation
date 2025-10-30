from typing import List

class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n
        
        for i, v in enumerate(nums):
            if v == 0:
                # If the value is 0, we just set result[i] to 0
                result[i] = 0
            else:
                # Compute the new index in the circular array
                new_index = (i + v) % n
                result[i] = nums[new_index]
        
        return result