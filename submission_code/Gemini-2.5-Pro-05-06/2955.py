class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # The problem requires rounding purchaseAmount to the nearest multiple of 10.
        # If purchaseAmount is exactly halfway between two multiples of 10 (e.g., 15),
        # it should be rounded to the larger multiple (e.g., 15 rounds to 20).
        # This behavior is "round half up".
        #
        # A standard integer arithmetic way to round `num` to the nearest multiple of `M`,
        # with halves rounded up (where M/2 is an integer), is: 
        # `((num + M_half) // M) * M`.
        # Here, M = 10, so M_half = 5.
        # Formula: `roundedAmount = ((purchaseAmount + 5) // 10) * 10`.
        
        # Calculate the amount to be paid after rounding.
        roundedAmount = ((purchaseAmount + 5) // 10) * 10
        
        # Initial bank account balance.
        initialBalance = 100
        
        # Calculate the balance after the purchase.
        accountBalanceAfterPurchase = initialBalance - roundedAmount
        
        return accountBalanceAfterPurchase