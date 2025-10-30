class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        single_digit_sum = 0
        double_digit_sum = 0
        total_sum = 0

        for num in nums:
            if num < 10:
                single_digit_sum += num
            else:
                double_digit_sum += num
            total_sum += num

        if single_digit_sum > total_sum - single_digit_sum:
            return True
        if double_digit_sum > total_sum - double_digit_sum:
            return True
        
        return False