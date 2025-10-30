class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Compute remainder to the nearest lower multiple of 10
        rem = purchaseAmount % 10
        
        # If remainder is less than 5, round down; otherwise round up (ties go up)
        if rem < 5:
            rounded = purchaseAmount - rem
        else:
            rounded = purchaseAmount + (10 - rem)
        
        # Initial balance is 100
        return 100 - rounded