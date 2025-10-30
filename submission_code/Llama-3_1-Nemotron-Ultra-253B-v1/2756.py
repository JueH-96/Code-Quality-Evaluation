from typing import List

class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        sorted_prices = sorted(prices)
        min_sum = sorted_prices[0] + sorted_prices[1]
        return money - min_sum if min_sum <= money else money