class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        n = len(nums)
        nums.sort()
        median_index = n // 2
        if n % 2 == 0:
            median = (nums[median_index - 1] + nums[median_index]) // 2
        else:
            median = nums[median_index]
        
        operations = 0
        for i in range(n):
            if nums[i] < k:
                operations += k - nums[i]
            elif nums[i] > k:
                operations += nums[i] - k
        
        return operations // 2