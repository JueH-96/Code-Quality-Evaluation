class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Round the purchaseAmount to the nearest multiple of 10
        # If equidistant, choose the higher multiple
        roundedAmount = ((purchaseAmount + 5) // 10) * 10
        # Calculate the new balance
        balance = 100 - roundedAmount
        return balance