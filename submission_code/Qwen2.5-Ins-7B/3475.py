from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        ones = sum(nums)
        if ones == n:
            return 0
        if ones == 0:
            return -1
        
        operations = 0
        i = 0
        while i < n - 2:
            if nums[i] == 0 and nums[i + 1] == 0 and nums[i + 2] == 0:
                operations += 1
                nums[i] = 1
                nums[i + 1] = 1
                nums[i + 2] = 1
                i += 3
            else:
                i += 1
        
        if sum(nums) != n:
            return -1
        return operations