class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Find the remainder when divided by 10
        remainder = purchaseAmount % 10
        
        # If remainder is 5 or more, round up; otherwise round down
        if remainder >= 5:
            roundedAmount = purchaseAmount + (10 - remainder)
        else:
            roundedAmount = purchaseAmount - remainder
        
        # Return the balance after purchase
        return 100 - roundedAmount