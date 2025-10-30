from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        required = set(range(1, k + 1))
        for m in range(1, len(nums) + 1):
            sub = nums[-m:]
            if set(sub) >= required:
                return m
        return len(nums)