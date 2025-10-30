from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        # For each element:
        # If remainder is 0 -> no operations.
        # If remainder is 1 or 2 -> exactly one operation is enough.
        operations = 0
        for num in nums:
            if num % 3 != 0:
                operations += 1
        return operations