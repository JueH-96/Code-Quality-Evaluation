class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        median_index = (n - 1) // 2
        median = nums[median_index]
        operations = 0
        if median == k:
            return 0
        elif median < k:
            operations += k - median
            for i in range(median_index + 1, n):
                if nums[i] < k:
                    operations += k - nums[i]
        else:
            operations += median - k
            for i in range(median_index):
                if nums[i] > k:
                    operations += nums[i] -k
        return operations