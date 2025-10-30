class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Calculate the nearest multiple of 10
        lower_multiple = (purchaseAmount // 10) * 10
        upper_multiple = lower_multiple + 10
        
        # Choose the larger multiple if both are equidistant
        if purchaseAmount - lower_multiple < upper_multiple - purchaseAmount:
            roundedAmount = lower_multiple
        else:
            roundedAmount = upper_multiple
        
        # Calculate the remaining balance
        account_balance = 100 - roundedAmount
        return account_balance