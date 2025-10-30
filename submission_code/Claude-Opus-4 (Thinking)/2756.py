class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        # Sort prices to find the two cheapest chocolates
        prices.sort()
        
        # Calculate the cost of the two cheapest chocolates
        total_cost = prices[0] + prices[1]
        
        # Check if we can afford them
        if total_cost <= money:
            return money - total_cost
        else:
            return money