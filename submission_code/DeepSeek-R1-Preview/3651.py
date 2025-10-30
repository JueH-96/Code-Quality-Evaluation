from typing import List

class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n
        for i in range(n):
            num = nums[i]
            if num > 0:
                new_i = (i + num) % n
                result[i] = nums[new_i]
            elif num < 0:
                steps = -num
                new_i = (i - steps) % n
                result[i] = nums[new_i]
            else:
                result[i] = 0
        return result