class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Round the purchaseAmount to the nearest multiple of 10
        lower_multiple = (purchaseAmount // 10) * 10
        upper_multiple = lower_multiple + 10
        
        if purchaseAmount - lower_multiple < upper_multiple - purchaseAmount:
            roundedAmount = lower_multiple
        else:
            roundedAmount = upper_multiple
        
        # Calculate the remaining balance
        initial_balance = 100
        remaining_balance = initial_balance - roundedAmount
        
        return remaining_balance