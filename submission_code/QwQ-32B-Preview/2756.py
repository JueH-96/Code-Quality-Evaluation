class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        prices.sort()
        sum_two_cheapest = prices[0] + prices[1]
        if sum_two_cheapest <= money:
            return money - sum_two_cheapest
        else:
            return money