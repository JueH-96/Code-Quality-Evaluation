class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        # Sort the prices to find the two cheapest chocolates
        prices.sort()
        
        # Calculate the sum of the two cheapest chocolates
        min_sum = prices[0] + prices[1]
        
        # Check if we can buy these two chocolates without going into debt
        if min_sum <= money:
            # Return the leftover money after buying the chocolates
            return money - min_sum
        else:
            # If we can't buy them, return the original amount of money
            return money