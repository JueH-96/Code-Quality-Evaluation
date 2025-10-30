class Solution:
    def minCost(self, nums: List[int]) -> int:
        cost = 0
        while len(nums) > 2:
            nums.sort()
            cost += max(nums[0], nums[1])
            nums.pop(0)
            nums.pop(0)
        if nums:
            cost += max(nums)
        return cost