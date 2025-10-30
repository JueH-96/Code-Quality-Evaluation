class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        """
        Calculates the account balance after a purchase with special rounding.

        The purchase amount is rounded to the nearest multiple of 10. If an amount
        is halfway between two multiples (e.g., 15), it is rounded up to the
        larger multiple (20).

        Args:
            purchaseAmount: The initial amount of the purchase.

        Returns:
            The remaining account balance after the rounded purchase.
        """
        
        # To round to the nearest multiple of 10 with halves rounded up,
        # we can add 5 to the purchaseAmount, perform integer division by 10,
        # and then multiply by 10.
        # For example, for purchaseAmount = 9:
        #   ((9 + 5) // 10) * 10  ->  (14 // 10) * 10  ->  1 * 10  ->  10
        # For purchaseAmount = 15:
        #   ((15 + 5) // 10) * 10 ->  (20 // 10) * 10  ->  2 * 10  ->  20
        
        roundedAmount = ((purchaseAmount + 5) // 10) * 10
        
        # The initial balance is 100. Subtract the rounded purchase amount.
        return 100 - roundedAmount