class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Initial balance
        initial_balance = 100
        
        # Round the purchase amount to the nearest multiple of 10
        roundedAmount = round(purchaseAmount / 10) * 10
        
        # Calculate the new balance after the purchase
        new_balance = initial_balance - roundedAmount
        
        return new_balance