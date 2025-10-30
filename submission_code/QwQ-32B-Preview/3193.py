from typing import List

class Solution:
    def maximumStrongPairXor(self, nums: List[int]) -> int:
        max_xor = 0
        for i in range(len(nums)):
            for j in range(len(nums)):
                x = nums[i]
                y = nums[j]
                if abs(x - y) <= min(x, y):
                    xor_value = x ^ y
                    if xor_value > max_xor:
                        max_xor = xor_value
        return max_xor