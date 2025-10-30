from typing import List

class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        sum_total = 0
        for i in range(len(nums)):
            if bin(i).count('1') == k:
                sum_total += nums[i]
        return sum_total