class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Round purchaseAmount to nearest multiple of 10
        # Add 5 before integer division to implement "round to nearest, ties up"
        rounded_amount = ((purchaseAmount + 5) // 10) * 10
        # Initial balance is 100 dollars, subtract the rounded purchase amount
        return 100 - rounded_amount