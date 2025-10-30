from typing import List

class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        # To minimize the sum of prices of two chocolates,
        # we must choose the two cheapest ones.
        # Sorting the prices list in ascending order makes this easy.
        prices.sort()
        
        # The two cheapest chocolates will be at prices[0] and prices[1]
        # after sorting.
        cost_of_two_cheapest = prices[0] + prices[1]
        
        # Check if we can afford these two chocolates.
        # The problem states that leftover money must be non-negative.
        if cost_of_two_cheapest <= money:
            # If affordable, return the money remaining after the purchase.
            return money - cost_of_two_cheapest
        else:
            # If even the two cheapest chocolates are too expensive (i.e., buying them
            # would result in debt), then we cannot buy two chocolates as per the
            # condition. In this scenario, we return the original amount of money.
            return money