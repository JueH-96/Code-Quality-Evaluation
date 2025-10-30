class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Round to nearest 10, with ties going to the larger value
        # Add 5 and then round down to nearest 10
        roundedAmount = ((purchaseAmount + 5) // 10) * 10
        return 100 - roundedAmount