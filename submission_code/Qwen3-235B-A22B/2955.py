class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        rounded = (purchaseAmount // 10) * 10
        if purchaseAmount % 10 >= 5:
            rounded += 10
        return 100 - rounded