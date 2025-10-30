class Solution:
    def minimumCoins(self, prices: list[int]) -> int:
        n = len(prices)
        min_coins = 0
        i = 0
        while i < n:
            min_coins += prices[i]
            i += prices[i] + 1
        return min_coins