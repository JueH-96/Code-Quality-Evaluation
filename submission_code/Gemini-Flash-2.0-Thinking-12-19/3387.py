class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        sorted_nums = sorted(nums)
        n = len(sorted_nums)
        median_index = (n - 1) // 2
        current_median = sorted_nums[median_index]
        operations = 0
        if current_median < k:
            for i in range(median_index, n):
                if sorted_nums[i] < k:
                    operations += k - sorted_nums[i]
        elif current_median > k:
            for i in range(median_index, -1, -1):
                if sorted_nums[i] > k:
                    operations += sorted_nums[i] - k
        return operations