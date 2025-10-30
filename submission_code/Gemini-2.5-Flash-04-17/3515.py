from typing import List

class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        """
        Determines if Alice can win the game by choosing either all single-digit
        or all double-digit numbers, such that her sum is strictly greater
        than the sum of Bob's numbers.

        Args:
            nums: A list of positive integers between 1 and 99.

        Returns:
            True if Alice can win, False otherwise.
        """
        sum_single = 0
        sum_double = 0

        # Iterate through the numbers and sum up single-digit and double-digit numbers separately
        for num in nums:
            if num >= 1 and num <= 9: # Check if the number is single-digit
                sum_single += num
            # Since the constraint is 1 <= nums[i] <= 99, any number not single-digit must be double-digit
            elif num >= 10 and num <= 99: # Check if the number is double-digit
                sum_double += num
            # No need for an else block as per constraints

        # Alice wins if the sum of her chosen numbers is strictly greater than
        # the sum of Bob's numbers.
        #
        # Strategy 1: Alice chooses single-digit numbers (sum_single).
        #             Bob gets double-digit numbers (sum_double).
        #             Alice wins if sum_single > sum_double.
        #
        # Strategy 2: Alice chooses double-digit numbers (sum_double).
        #             Bob gets single-digit numbers (sum_single).
        #             Alice wins if sum_double > sum_single.
        #
        # Alice can win if she wins with either strategy.
        # This means Alice wins if (sum_single > sum_double) or (sum_double > sum_single).
        # This is equivalent to checking if sum_single is NOT equal to sum_double.

        return sum_single != sum_double