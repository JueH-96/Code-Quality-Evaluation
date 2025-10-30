from typing import List

class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n
        for i in range(n):
            num = nums[i]
            if num == 0:
                result[i] = 0
            else:
                if num > 0:
                    new_idx = (i + num) % n
                else:
                    new_idx = (i - abs(num)) % n
                result[i] = nums[new_idx]
        return result