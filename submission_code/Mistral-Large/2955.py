class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Calculate the nearest multiple of 10
        if purchaseAmount % 10 == 0:
            roundedAmount = purchaseAmount
        else:
            remainder = purchaseAmount % 10
            if remainder < 5:
                roundedAmount = purchaseAmount - remainder
            else:
                roundedAmount = purchaseAmount + (10 - remainder)

        # Calculate the new account balance
        initial_balance = 100
        new_balance = initial_balance - roundedAmount

        return new_balance