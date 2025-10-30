from typing import List

class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n
        for i in range(n):
            if nums[i] > 0:
                new_i = (i + nums[i]) % n
            elif nums[i] < 0:
                new_i = (i - (-nums[i])) % n
            else:
                new_i = i
            result[i] = nums[new_i]
        return result