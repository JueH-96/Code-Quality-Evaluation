class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        sum_single = sum(num for num in nums if num < 10)
        sum_double = sum(num for num in nums if num >= 10)
        total = sum(nums)
        return sum_single > (total - sum_single) or sum_double > (total - sum_double)