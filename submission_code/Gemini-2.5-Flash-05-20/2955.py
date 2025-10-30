class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        # The initial bank account balance is 100 dollars.
        initial_balance = 100

        # Calculate the rounded amount based on the specified rules:
        # - Round to the nearest multiple of 10.
        # - If there's a tie (e.g., purchaseAmount ends in 5), choose the larger multiple.
        # This behavior is achieved by adding 5 to purchaseAmount,
        # then performing integer division by 10 (which truncates),
        # and finally multiplying by 10 to get the nearest multiple of 10.
        #
        # Examples:
        # purchaseAmount = 9:  (9 + 5) // 10 * 10 = 14 // 10 * 10 = 1 * 10 = 10
        # purchaseAmount = 15: (15 + 5) // 10 * 10 = 20 // 10 * 10 = 2 * 10 = 20
        # purchaseAmount = 4:  (4 + 5) // 10 * 10 = 9 // 10 * 10 = 0 * 10 = 0
        # purchaseAmount = 5:  (5 + 5) // 10 * 10 = 10 // 10 * 10 = 1 * 10 = 10
        # purchaseAmount = 100: (100 + 5) // 10 * 10 = 105 // 10 * 10 = 10 * 10 = 100
        
        rounded_amount = (purchaseAmount + 5) // 10 * 10

        # Calculate and return the final account balance.
        return initial_balance - rounded_amount