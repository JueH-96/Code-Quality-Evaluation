class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        prices.sort()
        if len(prices) < 2:
            return money
        choco_sum = prices[0] + prices[1]
        if choco_sum <= money:
            return money - choco_sum
        else:
            return money