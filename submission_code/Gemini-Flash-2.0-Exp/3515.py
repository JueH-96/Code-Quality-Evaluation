class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        single_digit_nums = [num for num in nums if num < 10]
        double_digit_nums = [num for num in nums if 10 <= num <= 99]
        
        sum_single = sum(single_digit_nums)
        sum_double = sum(double_digit_nums)
        
        remaining_single = [num for num in nums if num not in single_digit_nums]
        remaining_double = [num for num in nums if num not in double_digit_nums]
        
        sum_remaining_single = sum(remaining_single)
        sum_remaining_double = sum(remaining_double)
        
        if sum_single > sum_remaining_single:
            return True
        if sum_double > sum_remaining_double:
            return True
        
        return False