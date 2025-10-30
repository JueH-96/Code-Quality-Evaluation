from typing import List

class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        prices.sort()
        cheapest_two = prices[0] + prices[1]
        
        if cheapest_two <= money:
            return money - cheapest_two
        return money