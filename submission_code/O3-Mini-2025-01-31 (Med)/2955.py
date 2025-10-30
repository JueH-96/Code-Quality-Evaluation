class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Calculate the lower and upper multiples of 10
        lower = (purchaseAmount // 10) * 10
        upper = lower + 10
        
        # Compute differences
        diff_lower = purchaseAmount - lower
        diff_upper = upper - purchaseAmount
        
        # If the purchase amount is closer to upper or equally close, choose upper
        if diff_upper <= diff_lower:
            roundedAmount = upper
        else:
            roundedAmount = lower
        
        # Subtract the rounded purchase amount from the initial balance of 100
        return 100 - roundedAmount