class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Calculate the rounded amount
        roundedAmount = (purchaseAmount + 5) // 10 * 10
        
        # Calculate the remaining balance
        remainingBalance = 100 - roundedAmount
        
        return remainingBalance