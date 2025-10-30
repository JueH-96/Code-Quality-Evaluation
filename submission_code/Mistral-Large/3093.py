from typing import List

class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        def countSetBits(n: int) -> int:
            return bin(n).count('1')

        total_sum = 0
        for index, value in enumerate(nums):
            if countSetBits(index) == k:
                total_sum += value

        return total_sum