class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Calculate the rounded amount by adding 5 and then rounding down to the nearest multiple of 10
        roundedAmount = ((purchaseAmount + 5) // 10) * 10
        # Subtract the rounded amount from the initial balance
        return 100 - roundedAmount