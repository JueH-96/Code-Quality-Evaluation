from typing import List

class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        """
        Determines if Alice can win the game by choosing either all single-digit
        or all double-digit numbers.
        """
        
        # Sum of single-digit numbers (1-9)
        single_digit_sum = 0
        # Sum of double-digit numbers (10-99)
        double_digit_sum = 0
        
        # Iterate through the numbers to categorize and sum them.
        # The constraints guarantee 1 <= num <= 99.
        for num in nums:
            if num < 10:
                single_digit_sum += num
            else:
                double_digit_sum += num
                
        # Alice can win if she can pick a group (either single-digit or 
        # double-digit numbers) whose sum is strictly greater than the 
        # other group's sum.
        # This is possible if and only if the two sums are not equal.
        #
        # - If single_digit_sum > double_digit_sum, she picks single-digit numbers and wins.
        # - If double_digit_sum > single_digit_sum, she picks double-digit numbers and wins.
        #
        # If the sums are equal, she cannot win because her sum will not be *strictly* greater.
        
        return single_digit_sum != double_digit_sum