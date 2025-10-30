class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Round purchaseAmount to nearest multiple of 10, breaking ties upward.
        rounded = ((purchaseAmount + 5) // 10) * 10
        # Subtract the rounded cost from the initial balance of 100.
        return 100 - rounded