from typing import List

class Solution:
    def countPartitions(self, nums: List[int]) -> int:
        total = sum(nums)
        # If the total sum is even, every partition i (0 <= i < n-1) yields an even difference.
        # Otherwise, no partition can yield an even difference.
        return len(nums) - 1 if total % 2 == 0 else 0