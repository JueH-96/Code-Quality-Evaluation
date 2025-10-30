class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        roundedAmount = (purchaseAmount // 10) * 10 if purchaseAmount % 10 < 5 else (purchaseAmount // 10 + 1) * 10
        return 100 - roundedAmount