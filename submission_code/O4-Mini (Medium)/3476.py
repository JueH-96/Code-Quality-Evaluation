from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        operations = 0
        for x in nums:
            r = x % 3
            # Either subtract r times or add (3-r) times, take the minimum
            operations += min(r, 3 - r)
        return operations