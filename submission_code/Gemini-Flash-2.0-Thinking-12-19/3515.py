class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        single_digit_nums = []
        double_digit_nums = []
        for num in nums:
            if 1 <= num <= 9:
                single_digit_nums.append(num)
            elif 10 <= num <= 99:
                double_digit_nums.append(num)

        sum_single_digit = sum(single_digit_nums)
        sum_double_digit = sum(double_digit_nums)
        total_sum = sum(nums)

        sum_bob_if_single = total_sum - sum_single_digit
        sum_bob_if_double = total_sum - sum_double_digit

        if sum_single_digit > sum_bob_if_single or sum_double_digit > sum_bob_if_double:
            return True
        else:
            return False