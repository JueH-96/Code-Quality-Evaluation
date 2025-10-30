class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        div, rem = divmod(purchaseAmount, 10)
        if rem >= 5:
            div += 1
        rounded = div * 10
        return 100 - rounded