class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        sum_single = sum(x for x in nums if x < 10)
        sum_double = sum(x for x in nums if x >= 10)
        return sum_single != sum_double