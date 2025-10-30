class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        pizzas.sort()
        n = len(pizzas)
        m = n // 4
        res = 0
        i = n - 1
        for day in range(1, m + 1):
            if day % 2 == 1:
                res += pizzas[i]
                i -= 1
            else:
                res += pizzas[i - 1]
                i -= 2
        return res