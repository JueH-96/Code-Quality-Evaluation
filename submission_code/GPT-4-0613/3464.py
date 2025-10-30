class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        total = 0
        for i in range(len(nums)):
            total += nums[i] if i % 2 == 0 else -nums[i]
        return max(total, max(nums))