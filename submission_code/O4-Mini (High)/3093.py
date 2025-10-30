from typing import List

class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        total = 0
        for i, val in enumerate(nums):
            # count set bits in index i
            if bin(i).count('1') == k:
                total += val
        return total