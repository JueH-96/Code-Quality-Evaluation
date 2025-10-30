from typing import List

class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        # Sort the prices to get the two cheapest chocolates
        prices.sort()
        # Calculate the cost of the two cheapest chocolates
        cost = prices[0] + prices[1]
        # Check if the cost is within the budget
        if cost <= money:
            # Return the leftover money
            return money - cost
        else:
            # Return the initial money if it's not possible to buy two chocolates
            return money