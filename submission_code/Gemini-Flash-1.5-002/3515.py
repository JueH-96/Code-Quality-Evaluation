class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        single_digit_sum = 0
        double_digit_sum = 0
        other_sum = 0
        for num in nums:
            if num < 10:
                single_digit_sum += num
            elif num >=10 and num <=99:
                double_digit_sum += num
            else:
                other_sum += num

        total_sum = sum(nums)
        if single_digit_sum > total_sum - single_digit_sum:
            return True
        elif double_digit_sum > total_sum - double_digit_sum:
            return True
        else:
            return False