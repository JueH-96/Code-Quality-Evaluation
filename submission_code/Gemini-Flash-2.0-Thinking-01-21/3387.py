class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        sorted_nums = sorted(nums)
        n = len(nums)
        median_index = n // 2
        median_val = sorted_nums[median_index]

        if median_val == k:
            return 0
        elif median_val < k:
            operations = 0
            for i in range(median_index, n):
                if sorted_nums[i] < k:
                    operations += k - sorted_nums[i]
            return operations
        else: # median_val > k
            operations = 0
            for i in range(median_index + 1): # Include median index in the loop
                if sorted_nums[i] > k:
                    operations += sorted_nums[i] - k
            return operations