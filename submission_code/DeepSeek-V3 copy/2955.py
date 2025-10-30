class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Calculate the remainder when purchaseAmount is divided by 10
        remainder = purchaseAmount % 10
        # Determine the rounded amount
        if remainder < 5:
            roundedAmount = purchaseAmount - remainder
        else:
            roundedAmount = purchaseAmount + (10 - remainder)
        # Calculate the new balance
        balance = 100 - roundedAmount
        return balance