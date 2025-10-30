from typing import List

class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        sum_single_digits = 0
        sum_double_digits = 0

        for num in nums:
            # Constraints: 1 <= nums[i] <= 99.
            # All numbers are positive integers.
            # Single-digit numbers are in the range [1, 9].
            # Double-digit numbers are in the range [10, 99].
            
            if num < 10:  # This correctly categorizes numbers 1-9 as single-digit.
                sum_single_digits += num
            else:  # This correctly categorizes numbers 10-99 as double-digit.
                sum_double_digits += num
        
        # Alice has two options for choosing her numbers:
        # 1. Alice takes all single-digit numbers.
        #    In this case, Alice's sum is sum_single_digits.
        #    Bob gets the remaining numbers (all double-digit numbers), so Bob's sum is sum_double_digits.
        #    Alice wins if sum_single_digits > sum_double_digits.
        
        # 2. Alice takes all double-digit numbers.
        #    In this case, Alice's sum is sum_double_digits.
        #    Bob gets the remaining numbers (all single-digit numbers), so Bob's sum is sum_single_digits.
        #    Alice wins if sum_double_digits > sum_single_digits.
        
        # Alice wins the game if she can win by making *either* of these choices.
        # So, Alice wins if (sum_single_digits > sum_double_digits) OR (sum_double_digits > sum_single_digits).
        
        # This logical OR condition is true if and only if sum_single_digits is not equal to sum_double_digits.
        
        # If sum_single_digits == sum_double_digits:
        #   - If Alice picks single-digit numbers, her sum equals Bob's sum. Not strictly greater.
        #   - If Alice picks double-digit numbers, her sum equals Bob's sum. Not strictly greater.
        #   In this scenario, Alice cannot win.
        
        # If sum_single_digits != sum_double_digits:
        #   - If sum_single_digits > sum_double_digits, Alice can choose single-digit numbers and win.
        #   - If sum_double_digits > sum_single_digits, Alice can choose double-digit numbers and win.
        #   In this scenario, Alice can always make a choice to win.
            
        return sum_single_digits != sum_double_digits