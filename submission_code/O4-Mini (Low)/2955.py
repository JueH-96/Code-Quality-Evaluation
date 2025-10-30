class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # Round purchaseAmount to the nearest multiple of 10, breaking ties upward.
        # Adding 5 and integer‐dividing by 10 does exactly that:
        rounded = ((purchaseAmount + 5) // 10) * 10
        # Subtract from the initial balance of 100
        return 100 - rounded