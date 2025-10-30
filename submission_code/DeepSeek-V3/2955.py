class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Calculate the remainder when purchaseAmount is divided by 10
        remainder = purchaseAmount % 10
        if remainder < 5:
            roundedAmount = purchaseAmount - remainder
        else:
            roundedAmount = purchaseAmount + (10 - remainder)
        # Ensure that the roundedAmount is not negative
        roundedAmount = max(roundedAmount, 0)
        # Calculate the new balance
        balance = 100 - roundedAmount
        return balance