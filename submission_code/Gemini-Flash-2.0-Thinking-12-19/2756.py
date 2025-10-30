class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        prices.sort()
        choco_price = prices[0] + prices[1]
        if choco_price <= money:
            return money - choco_price
        else:
            return money