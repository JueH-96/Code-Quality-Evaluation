class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        single_digit_nums = []
        double_digit_nums = []
        for num in nums:
            if num < 10:
                single_digit_nums.append(num)
            else:
                double_digit_nums.append(num)
        
        single_digit_sum = sum(single_digit_nums)
        double_digit_sum = sum(double_digit_nums)
        
        remaining_sum_single = sum(double_digit_nums)
        remaining_sum_double = sum(single_digit_nums)
        
        if single_digit_sum > remaining_sum_single:
            return True
        if double_digit_sum > remaining_sum_double:
            return True
        
        return False