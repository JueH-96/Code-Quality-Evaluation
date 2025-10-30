from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        operations = 0
        n = len(nums)
        for i in range(n - 2):
            if nums[i] == 0:
                # Flip the triplet
                nums[i] ^= 1
                nums[i + 1] ^= 1
                nums[i + 2] ^= 1
                operations += 1
        # Check the last two elements
        if nums[-2] == 0 or nums[-1] == 0:
            return -1
        else:
            return operations