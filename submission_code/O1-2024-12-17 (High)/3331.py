from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        operations = 0
        while min(nums) < k:
            nums.remove(min(nums))
            operations += 1
        return operations