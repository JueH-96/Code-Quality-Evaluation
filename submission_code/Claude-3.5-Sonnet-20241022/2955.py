class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Find the lower and upper multiples of 10
        lower = (purchaseAmount // 10) * 10
        upper = lower + 10
        
        # Calculate differences from purchase amount
        diff_lower = abs(purchaseAmount - lower)
        diff_upper = abs(purchaseAmount - upper)
        
        # If difference is equal, choose upper multiple
        # If upper difference is less or equal, choose upper multiple
        if diff_lower > diff_upper or diff_lower == diff_upper:
            rounded = upper
        else:
            rounded = lower
            
        # Return remaining balance after purchase
        return 100 - rounded