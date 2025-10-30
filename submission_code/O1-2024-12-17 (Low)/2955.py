class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Determine the nearest multiple of 10 to purchaseAmount
        # In case of a tie, rounding is to the larger multiple.
        rounded_amount = ((purchaseAmount + 5) // 10) * 10
        
        # Initial balance is 100
        # Subtract the rounded_amount from this balance
        return 100 - rounded_amount