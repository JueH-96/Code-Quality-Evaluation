class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        median_index = n // 2
        operations = 0
        
        # Elements to the left of median should be <= k
        for i in range(median_index):
            if nums[i] > k:
                operations += nums[i] - k
        
        # Median element should be k
        operations += abs(nums[median_index] - k)
        
        # Elements to the right of median should be >= k
        for i in range(median_index + 1, n):
            if nums[i] < k:
                operations += k - nums[i]
        
        return operations