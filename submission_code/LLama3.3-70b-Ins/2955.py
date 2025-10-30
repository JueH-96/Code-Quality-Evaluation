class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Calculate the remainder of the purchase amount when divided by 10
        remainder = purchaseAmount % 10
        
        # If the remainder is less than 5, round down to the nearest multiple of 10
        if remainder < 5:
            roundedAmount = purchaseAmount - remainder
        # If the remainder is 5 or more, round up to the nearest multiple of 10
        else:
            roundedAmount = purchaseAmount + (10 - remainder)
        
        # Calculate the new account balance
        newBalance = 100 - roundedAmount
        
        return newBalance