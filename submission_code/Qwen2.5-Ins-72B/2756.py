from typing import List

class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        # Sort the prices to find the two cheapest chocolates
        prices.sort()
        
        # Check if we can buy the two cheapest chocolates
        if prices[0] + prices[1] <= money:
            return money - (prices[0] + prices[1])
        else:
            return money