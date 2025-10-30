from typing import List

class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        single_digit = [num for num in nums if num < 10]
        double_digit = [num for num in nums if num >= 10]
        
        sum_single = sum(single_digit)
        sum_double = sum(double_digit)
        
        sum_total = sum(nums)
        
        # If Alice chooses single-digit numbers, Bob gets the rest
        alice_sum = sum_single
        bob_sum = sum_total - alice_sum
        if alice_sum > bob_sum:
            return True
        
        # If Alice chooses double-digit numbers, Bob gets the rest
        alice_sum = sum_double
        bob_sum = sum_total - alice_sum
        if alice_sum > bob_sum:
            return True
        
        return False