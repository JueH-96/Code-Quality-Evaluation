from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        total_ops = 0
        for num in nums:
            remainder = num % 3
            ops = min(remainder, 3 - remainder)
            total_ops += ops
        return total_ops