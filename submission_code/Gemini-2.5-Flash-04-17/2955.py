class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # The initial bank account balance is 100 dollars.
        initial_balance = 100

        # The purchase amount is rounded to the nearest multiple of 10.
        # The rounding rule is:
        # 1. Find the nearest multiple of 10.
        # 2. If there is a tie (e.g., for 15, both 10 and 20 are equally near),
        #    choose the larger multiple (20).
        # This behavior is equivalent to standard rounding to the nearest 10,
        # where .5 is rounded up.
        # For a non-negative number `x`, rounding to the nearest multiple of `m` with
        # ties rounding up can be calculated using integer division as `(x + m // 2) // m * m`.
        # Here, `m = 10`. So, `m // 2 = 5`.
        # The rounded amount is calculated as (purchaseAmount + 5) // 10 * 10.

        roundedAmount = (purchaseAmount + 5) // 10 * 10

        # The account balance after the purchase is the initial balance minus the rounded amount.
        final_balance = initial_balance - roundedAmount

        return final_balance