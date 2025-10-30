from typing import List

class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        single_digit_nums = [num for num in nums if 1 <= num <= 9]
        double_digit_nums = [num for num in nums if 10 <= num <= 99]

        # Scenario 1: Alice chooses single-digit numbers
        alice_sum_single = sum(single_digit_nums)
        bob_sum_single = sum(double_digit_nums)
        if alice_sum_single > bob_sum_single:
            return True

        # Scenario 2: Alice chooses double-digit numbers
        alice_sum_double = sum(double_digit_nums)
        bob_sum_double = sum(single_digit_nums)
        if alice_sum_double > bob_sum_double:
            return True

        return False