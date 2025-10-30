class Solution:
    def isGood(self, nums: List[int]) -> bool:
        n = len(nums) - 1
        if nums.count(n) != 2:
            return False
        return sorted(nums[:-1]) == list(range(1, n))