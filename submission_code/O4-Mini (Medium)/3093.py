from typing import List

class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        total = 0
        for i, val in enumerate(nums):
            # Count the number of 1's in the binary representation of i
            if bin(i).count('1') == k:
                total += val
        return total