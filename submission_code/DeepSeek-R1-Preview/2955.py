class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        remainder = purchaseAmount % 10
        rounded = (purchaseAmount // 10) * 10
        if remainder >= 5:
            rounded += 10
        return 100 - rounded