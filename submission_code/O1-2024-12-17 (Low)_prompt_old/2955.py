class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Round to the nearest multiple of 10
        # If there is a tie, pick the larger multiple of 10
        rounded_amount = ((purchaseAmount + 5) // 10) * 10
        
        # Subtract the rounded purchase amount from the initial balance
        return 100 - rounded_amount