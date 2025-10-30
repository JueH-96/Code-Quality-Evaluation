class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        roundedAmount = round(purchaseAmount / 10) * 10
        return 100 - roundedAmount