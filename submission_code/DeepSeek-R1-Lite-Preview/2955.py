class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        quotient = purchaseAmount // 10
        remainder = purchaseAmount % 10
        if remainder < 5:
            roundedAmount = quotient * 10
        else:
            roundedAmount = (quotient + 1) * 10
        balance = 100 - roundedAmount
        return balance