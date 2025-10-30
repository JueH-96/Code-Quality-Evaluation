from typing import List

class Solution:
    def countPartitions(self, nums: List[int]) -> int:
        # If the total sum is odd, no partition can make left and right sums
        # have the same parity, hence no valid partitions exist.
        total = sum(nums)
        if total & 1:
            return 0
        # When the total sum is even, every possible partition (n - 1 of them)
        # yields left and right subarrays with equal parity, so all are valid.
        return len(nums) - 1