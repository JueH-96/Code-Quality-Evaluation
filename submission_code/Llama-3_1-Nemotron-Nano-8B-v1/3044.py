from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        required = set(range(1, k + 1))
        n = len(nums)
        for m in range(1, n + 1):
            current = set(nums[-m:])
            if required.issubset(current):
                return m
        return n