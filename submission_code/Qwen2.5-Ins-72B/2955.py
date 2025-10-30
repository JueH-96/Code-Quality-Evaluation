class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Calculate the nearest multiple of 10
        roundedAmount = (purchaseAmount + 5) // 10 * 10
        # Calculate the new account balance
        return 100 - roundedAmount