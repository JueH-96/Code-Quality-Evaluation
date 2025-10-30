class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        # Sort the prices to easily find the two smallest
        prices.sort()
        
        # Get the two smallest prices
        min_cost = prices[0] + prices[1]
        
        # Check if we can afford the two cheapest chocolates
        if min_cost <= money:
            return money - min_cost
        else:
            return money