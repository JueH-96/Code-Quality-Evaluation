from typing import List

class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        prices.sort()  # sort the prices to get the cheapest chocolates
        # Sum the two cheapest chocolates
        cost = prices[0] + prices[1]
        # If you can buy them with your money, return the leftover; otherwise, return the original money.
        return money - cost if cost <= money else money