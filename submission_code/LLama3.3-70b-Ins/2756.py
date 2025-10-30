from typing import List

class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        # Sort the prices in ascending order
        prices.sort()
        
        # Check if we can buy two chocolates without going into debt
        if prices[0] + prices[1] <= money:
            # If we can, return the leftover money
            return money - prices[0] - prices[1]
        else:
            # If we can't, return the initial amount of money
            return money