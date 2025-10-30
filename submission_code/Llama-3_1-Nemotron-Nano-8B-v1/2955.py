class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        rem = purchaseAmount % 10
        if rem < 5:
            rounded = purchaseAmount - rem
        else:
            rounded = purchaseAmount + (10 - rem)
        return 100 - rounded