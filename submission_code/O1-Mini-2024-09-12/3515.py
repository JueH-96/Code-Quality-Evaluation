class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        total = sum(nums)
        sum_single = sum(x for x in nums if x < 10)
        sum_double = sum(x for x in nums if x >= 10)
        return sum_single > (total - sum_single) or sum_double > (total - sum_double)