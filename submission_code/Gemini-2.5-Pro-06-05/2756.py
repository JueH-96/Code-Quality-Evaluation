from typing import List

class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        """
        Buys two chocolates with the minimum possible cost and returns the leftover money.

        The method finds the two cheapest chocolates in a single pass through the prices list.
        It then checks if their combined cost is within the budget. If it is, the leftover
        money is returned. Otherwise, the original amount of money is returned.
        """
        # Initialize two variables to track the two smallest prices.
        # float('inf') is used to ensure any price from the list will be smaller.
        min1 = float('inf')
        min2 = float('inf')

        # Iterate through the prices to find the two smallest values.
        for price in prices:
            if price < min1:
                # Found a new smallest value. The old smallest becomes the second smallest.
                min2 = min1
                min1 = price
            elif price < min2:
                # Found a new second smallest value.
                min2 = price

        # Calculate the sum of the two cheapest chocolates.
        cost = min1 + min2

        # If the cost is within the budget, return the leftover money.
        if cost <= money:
            return money - cost
        
        # Otherwise, it's impossible to buy two chocolates, so return the original money.
        return money