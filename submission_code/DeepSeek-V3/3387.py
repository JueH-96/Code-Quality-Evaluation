class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        median_index = n // 2
        operations = 0
        
        # Ensure the median is k
        if nums[median_index] < k:
            for i in range(median_index, n):
                if nums[i] < k:
                    operations += k - nums[i]
        elif nums[median_index] > k:
            for i in range(median_index, -1, -1):
                if nums[i] > k:
                    operations += nums[i] - k
        return operations