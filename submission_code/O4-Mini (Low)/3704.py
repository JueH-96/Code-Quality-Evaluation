from typing import List

class Solution:
    def countPartitions(self, nums: List[int]) -> int:
        total = sum(nums)
        # A partition has an even difference iff total sum is even.
        return len(nums) - 1 if total % 2 == 0 else 0