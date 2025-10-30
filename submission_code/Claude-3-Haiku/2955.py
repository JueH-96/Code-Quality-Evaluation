class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Calculate the rounded amount
        roundedAmount = round(purchaseAmount, -1)
        
        # Calculate the account balance after the purchase
        accountBalance = 100 - roundedAmount
        
        return accountBalance