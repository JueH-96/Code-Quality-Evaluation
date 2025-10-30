class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        n = len(nums)
        nums.sort()

        if n % 2 == 1:
            median_index = n // 2
            median_value = nums[median_index]
            operations = abs(median_value - k)
        else:
            median_index = n // 2
            median_value = nums[median_index]
            operations = abs(median_value - k)

        return operations