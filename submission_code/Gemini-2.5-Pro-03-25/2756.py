import math
from typing import List

class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        """
        Finds the minimum cost to buy exactly two chocolates such that the cost
        is less than or equal to the available money. Returns the leftover money
        if such a purchase is possible, otherwise returns the original money.

        Args:
            prices: A list of integers representing the prices of chocolates.
            money: An integer representing the initial amount of money.

        Returns:
            The leftover money after buying the two cheapest possible chocolates
            if affordable, otherwise the original amount of money.
        """

        # Approach 1: Sorting
        # Sort the prices to easily find the two cheapest chocolates.
        # Time Complexity: O(N log N) due to sorting
        # Space Complexity: O(log N) or O(N) depending on sort implementation

        # prices.sort()
        # min_cost = prices[0] + prices[1]

        # if min_cost <= money:
        #     return money - min_cost
        # else:
        #     return money

        # Approach 2: Single Pass Iteration
        # Find the two smallest prices in a single pass without full sorting.
        # Time Complexity: O(N)
        # Space Complexity: O(1)

        # Initialize the two smallest prices found so far to infinity
        # or a value larger than any possible price (max price is 100).
        min1 = float('inf')
        min2 = float('inf')

        # Iterate through the prices
        for price in prices:
            # If the current price is smaller than the smallest found so far (min1)
            if price < min1:
                # The previous min1 becomes the new second smallest (min2)
                min2 = min1
                # The current price becomes the new smallest (min1)
                min1 = price
            # Else if the current price is smaller than the second smallest (min2)
            # (but not smaller than min1)
            elif price < min2:
                # The current price becomes the new second smallest (min2)
                min2 = price

        # Calculate the minimum cost using the two smallest prices found
        min_cost = min1 + min2

        # Check if the minimum cost is affordable
        if min_cost <= money:
            # If affordable, return the leftover money
            return money - min_cost
        else:
            # If not affordable, return the original amount of money
            return money