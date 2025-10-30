class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        lower_multiple = (purchaseAmount // 10) * 10
        higher_multiple = lower_multiple + 10
        dist_lower = abs(purchaseAmount - lower_multiple)
        dist_higher = abs(purchaseAmount - higher_multiple)
        if dist_lower <= dist_higher:
            roundedAmount = lower_multiple if dist_lower < dist_higher else higher_multiple
        else:
            roundedAmount = higher_multiple
        return 100 - roundedAmount