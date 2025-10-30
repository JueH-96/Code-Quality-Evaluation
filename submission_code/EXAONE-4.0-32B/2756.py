class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        sorted_prices = sorted(prices)
        total = sorted_prices[0] + sorted_prices[1]
        if total <= money:
            return money - total
        else:
            return money