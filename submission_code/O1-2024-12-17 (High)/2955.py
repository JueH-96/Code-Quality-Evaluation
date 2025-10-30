class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        remainder = purchaseAmount % 10
        lower_multiple = purchaseAmount - remainder
        higher_multiple = lower_multiple + 10
        
        # Compare differences to the lower and higher multiples of 10
        diff_to_lower = remainder
        diff_to_higher = 10 - remainder
        
        # If there's a tie or the higher multiple is closer, choose the higher multiple
        if diff_to_higher <= diff_to_lower:
            pay = higher_multiple
        else:
            pay = lower_multiple
        
        return 100 - pay