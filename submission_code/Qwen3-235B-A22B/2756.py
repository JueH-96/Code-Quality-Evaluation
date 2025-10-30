from typing import List

class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        prices.sort()
        min_sum = prices[0] + prices[1]
        if min_sum <= money:
            return money - min_sum
        else:
            return money