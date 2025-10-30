class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        single_digit_sum = sum(filter(lambda x: 1 <= x <= 9, nums))
        double_digit_sum = sum(filter(lambda x: 10 <= x <= 99, nums))
        return single_digit_sum > double_digit_sum or double_digit_sum > single_digit_sum