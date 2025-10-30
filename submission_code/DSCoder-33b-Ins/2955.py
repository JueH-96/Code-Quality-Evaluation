class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        balance = 100
        roundedAmount = round(purchaseAmount, -1)
        balance -= roundedAmount
        return balance