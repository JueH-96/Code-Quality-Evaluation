from typing import List

class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        # Sort the prices to get the two smallest prices
        prices.sort()
        total_cost = prices[0] + prices[1]
        # If we can afford both, return leftover money; otherwise return the original money
        if total_cost <= money:
            return money - total_cost
        return money