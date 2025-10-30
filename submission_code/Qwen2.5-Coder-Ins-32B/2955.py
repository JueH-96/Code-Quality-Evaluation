class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Calculate the nearest multiple of 10
        roundedAmount = (purchaseAmount + 5) // 10 * 10
        # Return the remaining balance after the purchase
        return 100 - roundedAmount