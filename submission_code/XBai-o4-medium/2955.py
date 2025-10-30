class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        remainder = purchaseAmount % 10
        if remainder < 5:
            roundedAmount = (purchaseAmount // 10) * 10
        else:
            roundedAmount = (purchaseAmount // 10 + 1) * 10
        return 100 - roundedAmount