from typing import List

class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        # Sort the prices array to find the two smallest prices easily.
        # This takes O(N log N) time, where N is the number of prices.
        # Given the constraints (N <= 50), this is highly efficient.
        prices.sort()

        # The two smallest prices are now at the beginning of the sorted array.
        # We are guaranteed that prices.length >= 2 by the constraints,
        # so prices[0] and prices[1] are valid accesses.
        cheapest1 = prices[0]
        cheapest2 = prices[1]

        # Calculate the total cost of buying the two cheapest chocolates.
        cost = cheapest1 + cheapest2

        # Check if the total cost is less than or equal to the available money.
        # This condition ensures that the leftover money (money - cost) is non-negative.
        if money >= cost:
            # If the cost is affordable, return the amount of money left over.
            return money - cost
        else:
            # If the cost of the two cheapest chocolates is greater than the available money,
            # it's impossible to buy any two chocolates without ending up with negative money.
            # According to the problem statement, if there is no way to buy two chocolates
            # without ending up in debt (i.e., with non-negative leftover money),
            # we should return the original amount of money.
            return money