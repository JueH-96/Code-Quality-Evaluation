class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        total_cost = 0
        for i in range(n):
            total_cost += prices[i]
            if i + 1 < n:
                total_cost -= min(prices[i+1:i+1+i])
        return total_cost