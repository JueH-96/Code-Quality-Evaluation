class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        prices.sort()
        total_coins = 0
        for price in prices:
            total_coins += price
        return total_coins