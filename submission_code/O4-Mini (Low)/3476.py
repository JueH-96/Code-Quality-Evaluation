from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        ops = 0
        for x in nums:
            r = x % 3
            if r == 1:
                ops += 1   # best to subtract 1
            elif r == 2:
                ops += 1   # best to add 1
        return ops