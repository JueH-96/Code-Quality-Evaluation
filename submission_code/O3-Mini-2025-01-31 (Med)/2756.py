from typing import List

class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        # Sort the prices to get the two cheapest chocolates
        prices.sort()
        # Sum the two smallest prices
        total_cost = prices[0] + prices[1]
        # Check if we can afford these two chocolates without going into debt
        if total_cost > money:
            return money
        # Return the leftover money after the purchase
        return money - total_cost