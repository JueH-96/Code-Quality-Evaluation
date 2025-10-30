from typing import List

class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        # Find the two smallest prices
        prices.sort()
        cost = prices[0] + prices[1]
        
        # Check if we can buy the two chocolates without going in debt
        if cost <= money:
            return money - cost
        else:
            return money