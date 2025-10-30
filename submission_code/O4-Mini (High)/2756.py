from typing import List

class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        # Sort prices to get the two cheapest chocolates
        prices.sort()
        cheapest_sum = prices[0] + prices[1]
        # If we can afford them, return the leftover money, otherwise return original money
        if cheapest_sum <= money:
            return money - cheapest_sum
        return money