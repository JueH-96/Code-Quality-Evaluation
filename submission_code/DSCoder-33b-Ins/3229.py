class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        mid = nums[n // 2]
        return sum(abs(num - mid) for num in nums)