class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        min_cost = float('inf')
        found = False
        n = len(prices)
        for i in range(n):
            for j in range(i + 1, n):
                cost = prices[i] + prices[j]
                if cost <= money:
                    min_cost = min(min_cost, cost)
                    found = True

        if found:
            return money - min_cost
        else:
            return money