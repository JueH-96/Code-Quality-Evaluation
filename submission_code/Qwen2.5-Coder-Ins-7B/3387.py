class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        median_index = n // 2
        median = nums[median_index]
        
        if median == k:
            return 0
        
        operations = 0
        if median < k:
            for i in range(median_index, n):
                operations += max(0, nums[i] - k)
        else:
            for i in range(median_index, -1, -1):
                operations += max(0, k - nums[i])
        
        return operations