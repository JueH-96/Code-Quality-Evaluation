class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        
        # Find median index - for even length, we take the larger median
        median_idx = n // 2
        
        operations = 0
        
        # Change the median to k
        operations += abs(nums[median_idx] - k)
        
        # For elements before median: if they're greater than k, decrease them to k
        for i in range(median_idx):
            if nums[i] > k:
                operations += nums[i] - k
        
        # For elements after median: if they're less than k, increase them to k
        for i in range(median_idx + 1, n):
            if nums[i] < k:
                operations += k - nums[i]
        
        return operations