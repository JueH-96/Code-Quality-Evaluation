import heapq

class Solution:
    def minOperationsToMakeMedianK(self, nums: list[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        median_index = (n - 1) // 2
        median = nums[median_index]
        
        operations = 0
        if median < k:
            operations = k - median
            
        elif median > k:
            operations = median - k
        
        return operations