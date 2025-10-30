from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        operations = 0
        for num in nums:
            operations += min(num % 3, 3 - num % 3)
        return operations