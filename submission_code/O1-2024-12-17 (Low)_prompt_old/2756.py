class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        prices.sort()
        # Sum of the two cheapest chocolates
        total_cost = prices[0] + prices[1]
        # Check if we have enough money for these two chocolates
        if total_cost <= money:
            return money - total_cost
        else:
            return money