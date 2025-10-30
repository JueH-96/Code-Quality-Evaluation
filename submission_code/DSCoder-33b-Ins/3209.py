class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        prices.sort(reverse=True)
        total_coins = 0
        for i in range(len(prices)):
            if i % 2 == 0:
                total_coins += prices[i]
        return total_coins