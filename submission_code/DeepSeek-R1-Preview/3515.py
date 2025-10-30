class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        sum_single = 0
        sum_double = 0
        for num in nums:
            if num < 10:
                sum_single += num
            else:
                sum_double += num
        total = sum_single + sum_double
        return sum_single * 2 > total or sum_double * 2 > total