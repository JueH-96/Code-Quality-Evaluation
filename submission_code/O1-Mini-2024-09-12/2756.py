class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        prices.sort()
        sum_two = prices[0] + prices[1]
        if sum_two <= money:
            return money - sum_two
        else:
            return money