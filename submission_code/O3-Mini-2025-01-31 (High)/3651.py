from typing import List

class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n
        for i, num in enumerate(nums):
            # For nums[i] == 0, result[i] is just nums[i]
            if num == 0:
                result[i] = num
            else:
                # Using circular index arithmetic for both positive and negative values
                new_index = (i + num) % n
                result[i] = nums[new_index]
        return result