class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        if purchaseAmount == 0:
            return 100
        remainder = purchaseAmount % 10
        if remainder <= 5:
            roundedAmount = purchaseAmount - remainder
        else:
            roundedAmount = purchaseAmount + (10 - remainder)
        return 100 - roundedAmount