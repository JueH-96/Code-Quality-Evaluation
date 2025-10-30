class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        lower_multiple = (purchaseAmount // 10) * 10
        upper_multiple = lower_multiple + 10

        diff_lower = abs(purchaseAmount - lower_multiple)
        diff_upper = abs(purchaseAmount - upper_multiple)

        if diff_lower < diff_upper:
            roundedAmount = lower_multiple
        elif diff_upper < diff_lower:
            roundedAmount = upper_multiple
        else:
            roundedAmount = upper_multiple

        account_balance = 100 - roundedAmount
        return account_balance