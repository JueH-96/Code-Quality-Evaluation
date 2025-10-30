from typing import List

class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n

        for i in range(n):
            if nums[i] > 0:
                result[i] = (i + nums[i]) % n
            elif nums[i] < 0:
                result[i] = (i - abs(nums[i])) % n
            else:
                result[i] = nums[i]

        return result