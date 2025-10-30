from typing import List

class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        median_index = n // 2
        median = nums[median_index]
        operations = 0
        
        if median == k:
            return 0
        
        if median < k:
            for i in range(median_index, n):
                if nums[i] < k:
                    operations += k - nums[i]
                else:
                    break
        else:
            for i in range(median_index, -1, -1):
                if nums[i] > k:
                    operations += nums[i] - k
                else:
                    break
        
        return operations