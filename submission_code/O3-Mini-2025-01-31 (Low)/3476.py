from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        operations = 0
        for num in nums:
            remainder = num % 3
            # Minimal operations is the minimum between remainder and (3 - remainder)
            # For remainder 0, it yields 0; for 1 it yields 1; for 2 it also yields 1.
            operations += min(remainder, 3 - remainder)
        return operations