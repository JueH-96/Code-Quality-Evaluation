class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        # Sort the array to find the two cheapest chocolates
        prices.sort()
        
        # Calculate the total cost of the two cheapest chocolates
        cost = prices[0] + prices[1]
        
        # If you can afford these two chocolates, return leftover money
        if cost <= money:
            return money - cost
        
        # If not, you cannot buy two chocolates without going into debt, so return the original money
        return money