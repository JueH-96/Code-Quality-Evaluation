class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        balance = 100
        rounded_amount = round(purchaseAmount, -1)
        if rounded_amount < purchaseAmount:
            rounded_amount += 10
        return balance - rounded_amount