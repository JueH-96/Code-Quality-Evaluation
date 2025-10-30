class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        rounded_amount = round(purchaseAmount / 10) * 10
        if purchaseAmount % 10 == 5:
            rounded_amount = (purchaseAmount // 10 + 1) * 10
        
        return 100 - rounded_amount