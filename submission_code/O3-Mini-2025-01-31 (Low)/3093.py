from typing import List

class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        total = 0
        for idx, num in enumerate(nums):
            # count the set bits in idx using built-in bin function
            if bin(idx).count('1') == k:
                total += num
        return total