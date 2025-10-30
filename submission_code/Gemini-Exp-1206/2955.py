class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        lower = (purchaseAmount // 10) * 10
        upper = lower + 10
        if purchaseAmount - lower < upper - purchaseAmount:
            roundedAmount = lower
        elif purchaseAmount - lower > upper - purchaseAmount:
            roundedAmount = upper
        else:
            roundedAmount = upper
        return 100 - roundedAmount