class Solution:
    def getFinalState(self, nums, k, multiplier):
        for _ in range(k):
            min_index = nums.index(min(nums))
            nums[min_index] *= multiplier
        return nums