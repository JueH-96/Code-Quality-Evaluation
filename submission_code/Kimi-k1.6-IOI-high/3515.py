class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        total = sum(nums)
        sum_single = sum(x for x in nums if x < 10)
        sum_double = sum(x for x in nums if x >= 10)
        return sum_single * 2 > total or sum_double * 2 > total