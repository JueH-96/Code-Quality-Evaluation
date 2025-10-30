class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        total_sum = sum(nums)
        sum_single_digit = sum(n for n in nums if n < 10)
        sum_double_digit = sum(n for n in nums if n >= 10)
        if sum_single_digit > total_sum - sum_single_digit:
            return True
        if sum_double_digit > total_sum - sum_double_digit:
            return True
        return False