class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Calculate the nearest multiple of 10 to the purchase amount
        nearest_multiple = round(purchaseAmount / 10) * 10
        # Calculate the rounded amount by checking if the purchase amount is closer to the next multiple of 10
        rounded_amount = nearest_multiple + 10 if abs(nearest_multiple - purchaseAmount) >= abs(nearest_multiple + 10 - purchaseAmount) else nearest_multiple
        # Calculate the account balance after the purchase
        account_balance = 100 - rounded_amount
        return account_balance