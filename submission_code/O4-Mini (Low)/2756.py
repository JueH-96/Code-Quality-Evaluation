from typing import List

class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        # Sort prices to get the two smallest values
        prices.sort()
        # Sum of the two cheapest chocolates
        cost = prices[0] + prices[1]
        # If we can afford them, return leftover money; otherwise return original money
        return money - cost if cost <= money else money