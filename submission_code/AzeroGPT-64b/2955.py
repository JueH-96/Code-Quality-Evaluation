class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Find the nearest multiple of 10 by rounding the purchase amount
        nearestMultiple = round(purchaseAmount / 10) * 10
        # Adjust the nearest multiple if there's a tie between two multiples
        if abs(purchaseAmount - nearestMultiple) == 5:
            nearestMultiple += 10
        # Calculate the balance by subtracting the rounded purchase amount from 100
        balance = 100 - nearestMultiple
        return balance