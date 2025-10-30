from typing import List

class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        total_sum = 0
        for index, value in enumerate(nums):
            if bin(index).count('1') == k:
                total_sum += value
        return total_sum