class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        single_digit_sum = sum(i for i in nums if i < 10)
        double_digit_sum = sum(i for i in nums if i >= 10)
        return single_digit_sum != double_digit_sum