class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        mid = n // 2
        return sum(abs(nums[i] - nums[mid]) for i in range(n))