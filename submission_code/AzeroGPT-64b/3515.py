class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        single_digit_sum = sum(num for num in nums if 1 <= num <= 9)
        double_digit_sum = sum(num for num in nums if 10 <= num <= 99)
        
        return single_digit_sum > double_digit_sum or double_digit_sum > single_digit_sum - single_digit_sum * 2 if len(nums) > 1 else True