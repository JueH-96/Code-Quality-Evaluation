class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        # Sort the prices to find the two smallest values easily
        prices.sort()
        # Calculate the money left after buying the two smallest priced chocolates
        remaining = money - prices[0] - prices[1]
        # If the remaining money is negative, it means you can't afford two chocolates
        # so you return the initial amount of money
        if remaining < 0:
            return money
        # Otherwise, return the remaining money
        return remaining