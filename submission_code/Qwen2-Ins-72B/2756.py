class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        # Sort the prices to find the two smallest prices easily
        prices.sort()
        
        # Calculate the total cost of the two smallest priced chocolates
        total_cost = prices[0] + prices[1]
        
        # Check if the total cost is less than or equal to the money available
        if total_cost <= money:
            # Calculate and return the leftover money
            return money - total_cost
        else:
            # If it's not possible to buy two chocolates, return the initial money
            return money