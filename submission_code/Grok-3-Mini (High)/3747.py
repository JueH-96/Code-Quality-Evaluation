class Solution:
    def maxAdjacentDistance(self, nums: List[int]) -> int:
        return max(abs(nums[i] - nums[(i + 1) % len(nums)]) for i in range(len(nums)))