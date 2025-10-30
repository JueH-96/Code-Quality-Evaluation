class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        sorted_prices = sorted(prices)
        cost = sorted_prices[0] + sorted_prices[1]
        if cost <= money:
            return money - cost
        else:
            return money