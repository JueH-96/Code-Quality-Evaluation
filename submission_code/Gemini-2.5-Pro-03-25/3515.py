import math # This import is not necessary for the solution and can be removed.
from typing import List

class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        """
        Determines if Alice can win the game by choosing either all single-digit
        or all double-digit numbers such that her sum is strictly greater than Bob's sum.

        The game rules imply that if Alice chooses one category of numbers (e.g., single-digit),
        Bob receives the other category (e.g., double-digit). Alice wins if the sum of numbers
        she chose is strictly greater than the sum of numbers Bob received.

        Args:
            nums: A list of positive integers where 1 <= nums[i] <= 99.

        Returns:
            True if Alice can win by choosing either single-digit or double-digit numbers,
            False otherwise.
        """
        
        sum_single_digits = 0
        sum_double_digits = 0

        # Iterate through the list of numbers to calculate the sum of single-digit
        # numbers and the sum of double-digit numbers.
        for num in nums:
            # Check if the number is single-digit (1-9).
            # The constraint 1 <= nums[i] guarantees that num is positive.
            if num < 10: 
                sum_single_digits += num
            # Otherwise, the number must be double-digit (10-99).
            # The constraint nums[i] <= 99 guarantees this.
            else: 
                sum_double_digits += num
        
        # Determine if Alice can win.
        # Scenario 1: Alice chooses all single-digit numbers.
        # Alice's sum = sum_single_digits
        # Bob's sum = sum_double_digits
        # Alice wins if sum_single_digits > sum_double_digits.

        # Scenario 2: Alice chooses all double-digit numbers.
        # Alice's sum = sum_double_digits
        # Bob's sum = sum_single_digits
        # Alice wins if sum_double_digits > sum_single_digits.

        # Alice can win if either of these scenarios leads to her victory.
        # This is true if and only if the sum of single-digit numbers is not equal
        # to the sum of double-digit numbers. If the sums are equal, Alice cannot
        # achieve a strictly greater sum than Bob, regardless of her choice.
        
        return sum_single_digits != sum_double_digits