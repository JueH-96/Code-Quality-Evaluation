class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        median_idx = n // 2
        
        operations = 0
        
        # Elements before median should be <= k
        for i in range(median_idx):
            if nums[i] > k:
                operations += nums[i] - k
        
        # Median element should be k
        operations += abs(nums[median_idx] - k)
        
        # Elements after median should be >= k
        for i in range(median_idx + 1, n):
            if nums[i] < k:
                operations += k - nums[i]
        
        return operations