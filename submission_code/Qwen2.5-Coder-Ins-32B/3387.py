from typing import List

class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        median_index = n // 2
        median = nums[median_index]
        
        operations = 0
        if median < k:
            while median < k:
                operations += k - median
                median_index += 1
                if median_index >= n:
                    break
                median = nums[median_index]
        elif median > k:
            while median > k:
                operations += median - k
                median_index -= 1
                if median_index < 0:
                    break
                median = nums[median_index]
        
        return operations