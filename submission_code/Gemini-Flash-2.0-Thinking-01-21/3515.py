class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        single_digit_nums = []
        double_digit_nums = []
        for num in nums:
            if 1 <= num <= 9:
                single_digit_nums.append(num)
            elif 10 <= num <= 99:
                double_digit_nums.append(num)

        single_digit_sum = sum(single_digit_nums)
        double_digit_sum = sum(double_digit_nums)
        total_sum = sum(nums)

        bob_sum_scenario1 = total_sum - single_digit_sum
        if single_digit_sum > bob_sum_scenario1:
            return True

        bob_sum_scenario2 = total_sum - double_digit_sum
        if double_digit_sum > bob_sum_scenario2:
            return True

        return False