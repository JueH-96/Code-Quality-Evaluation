class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        # Sort the prices to easily find the two cheapest chocolates
        prices.sort()
        # Calculate the total cost of the two cheapest chocolates
        total = prices[0] + prices[1]
        # If the total cost is less than or equal to the money, return the leftover
        if total <= money:
            return money - total
        else:
            # Otherwise, return the original money
            return money