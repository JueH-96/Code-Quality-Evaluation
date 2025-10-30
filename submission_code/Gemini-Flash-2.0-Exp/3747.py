class Solution:
    def maxAdjacentDistance(self, nums: List[int]) -> int:
        max_diff = 0
        for i in range(len(nums)):
            if i == len(nums) - 1:
                diff = abs(nums[i] - nums[0])
            else:
                diff = abs(nums[i] - nums[i+1])
            if diff > max_diff:
                max_diff = diff
        return max_diff