class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        prices.sort()
        if len(prices) < 2:
            return money
        
        if prices[0] + prices[1] <= money:
            return money - (prices[0] + prices[1])
        else:
            return money