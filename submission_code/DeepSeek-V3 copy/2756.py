class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        # Sort the prices to easily find the two cheapest chocolates
        prices.sort()
        # Calculate the total cost of the two cheapest chocolates
        total_cost = prices[0] + prices[1]
        # If the total cost is less than or equal to money, return the leftover
        if total_cost <= money:
            return money - total_cost
        # Otherwise, return the original money
        else:
            return money