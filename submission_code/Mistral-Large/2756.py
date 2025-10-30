from typing import List

class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        # Sort the prices to find the two smallest values
        prices.sort()

        # Calculate the total cost of the two cheapest chocolates
        total_cost = prices[0] + prices[1]

        # Check if we can afford the two cheapest chocolates
        if total_cost <= money:
            return money - total_cost
        else:
            return money