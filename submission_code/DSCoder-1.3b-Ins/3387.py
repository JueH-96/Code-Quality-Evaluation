class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        median = nums[len(nums) // 2]
        operations = 0
        for num in nums:
            if num < median:
                operations += median - num
            elif num > median:
                operations += num - median
        return operations