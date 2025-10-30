class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # We first determine the remainder when dividing by 10
        remainder = purchaseAmount % 10
        
        # Calculate the lower multiple of 10 and the upper multiple of 10
        lower_multiple = purchaseAmount - remainder
        upper_multiple = lower_multiple + 10
        
        # If remainder is strictly less than 5, choose the lower multiple
        # If remainder is strictly greater than 5, choose the upper multiple
        # If remainder is exactly 5, we choose the upper multiple (tie -> choose larger)
        if remainder < 5:
            rounded_amount = lower_multiple
        else:
            rounded_amount = upper_multiple
        
        # Subtract the rounded amount from the initial balance (100)
        return 100 - rounded_amount