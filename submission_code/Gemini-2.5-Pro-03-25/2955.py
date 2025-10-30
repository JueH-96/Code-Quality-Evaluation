import math

class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        """
        Calculates the account balance after a purchase, where the purchase amount
        is rounded to the nearest multiple of 10 according to specific rules.

        The rounding rule is:
        1. Find the non-negative multiple of 10, roundedAmount, such that
           abs(roundedAmount - purchaseAmount) is minimized.
        2. If there are two nearest multiples of 10 (e.g., for purchaseAmount = 15,
           the nearest are 10 and 20), the larger multiple is chosen.

        This is equivalent to rounding to the nearest multiple of 10, with ties
        (numbers ending in 5) rounded up to the next higher multiple of 10.

        Args:
            purchaseAmount: An integer representing the amount to spend (0 <= purchaseAmount <= 100).

        Returns:
            An integer denoting the account balance after making the purchase
            with the rounded amount deducted from the initial balance of 100.
        """

        # Calculate the rounded amount.
        # The "round half up" strategy for multiples of 10 can be implemented
        # by adding 5 to the purchase amount, then performing integer division
        # by 10, and finally multiplying the result by 10.
        # Example:
        # purchaseAmount = 9 => (9 + 5) // 10 * 10 = 14 // 10 * 10 = 1 * 10 = 10
        # purchaseAmount = 15 => (15 + 5) // 10 * 10 = 20 // 10 * 10 = 2 * 10 = 20
        # purchaseAmount = 14 => (14 + 5) // 10 * 10 = 19 // 10 * 10 = 1 * 10 = 10
        # purchaseAmount = 4 => (4 + 5) // 10 * 10 = 9 // 10 * 10 = 0 * 10 = 0

        roundedAmount = (purchaseAmount + 5) // 10 * 10

        # The initial account balance is 100 dollars.
        initial_balance = 100

        # Calculate the final balance after deducting the rounded purchase amount.
        final_balance = initial_balance - roundedAmount

        return final_balance