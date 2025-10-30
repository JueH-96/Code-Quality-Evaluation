class Solution:
    def isGood(self, nums):
        n = max(nums)
        return nums.count(n) == 2 and sorted(nums) == sorted(list(range(1, n)) + [n, n])