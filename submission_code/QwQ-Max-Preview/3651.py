from typing import List

class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = []
        for i in range(n):
            current = nums[i]
            if current == 0:
                result.append(0)
            elif current > 0:
                new_i = (i + current) % n
                result.append(nums[new_i])
            else:
                steps = abs(current)
                new_i = (i - steps) % n
                result.append(nums[new_i])
        return result