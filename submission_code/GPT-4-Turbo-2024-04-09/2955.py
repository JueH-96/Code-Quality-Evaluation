class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Calculate the lower and upper multiples of 10
        lower_multiple = (purchaseAmount // 10) * 10
        upper_multiple = lower_multiple + 10
        
        # Determine the closest multiple of 10
        if (purchaseAmount - lower_multiple) < (upper_multiple - purchaseAmount):
            roundedAmount = lower_multiple
        else:
            roundedAmount = upper_multiple
        
        # Calculate the new account balance after the purchase
        new_balance = 100 - roundedAmount
        return new_balance