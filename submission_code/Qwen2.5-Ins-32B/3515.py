from typing import List

class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        single_digit_sum = sum(num for num in nums if num < 10)
        double_digit_sum = sum(num for num in nums if num >= 10)
        total_sum = sum(nums)
        
        # Bob's sum if Alice chooses single-digit numbers
        bob_sum_if_single = total_sum - single_digit_sum
        # Bob's sum if Alice chooses double-digit numbers
        bob_sum_if_double = total_sum - double_digit_sum
        
        # Alice wins if either of her choices results in a sum greater than Bob's
        return single_digit_sum > bob_sum_if_single or double_digit_sum > bob_sum_if_double