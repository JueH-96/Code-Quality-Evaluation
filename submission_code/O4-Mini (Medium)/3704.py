from typing import List

class Solution:
    def countPartitions(self, nums: List[int]) -> int:
        total_sum = sum(nums)
        # If total sum is even, all possible partitions (n-1 of them) have even difference.
        # Otherwise, none do.
        return len(nums) - 1 if total_sum % 2 == 0 else 0