class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        single_digit_nums = []
        double_digit_nums = []
        for num in nums:
            if num < 10:
                single_digit_nums.append(num)
            else:
                double_digit_nums.append(num)
        
        sum_single = sum(single_digit_nums)
        sum_double = sum(double_digit_nums)
        
        if sum_single > sum_double or sum_double > sum_single:
            return True
        else:
            return False