class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        n = len(nums)
        nums.sort()

        # Find the median index
        if n % 2 == 0:
            median_index = n // 2
        else:
            median_index = n // 2

        # Calculate the operations needed to make the median equal to k
        operations = 0
        for i in range(median_index, n):
            if nums[i] < k:
                operations += k - nums[i]
            elif nums[i] > k:
                operations += nums[i] - k

        return operations