class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        prices.sort()
        min_total_price = prices[0] + prices[1]
        if min_total_price <= money:
            return money - min_total_price
        else:
            return money