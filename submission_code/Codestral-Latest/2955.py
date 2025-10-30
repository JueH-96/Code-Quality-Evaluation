class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Calculate the nearest multiple of 10
        roundedAmount = (purchaseAmount + 5) // 10 * 10

        # Calculate the remaining balance
        remainingBalance = 100 - roundedAmount

        return remainingBalance