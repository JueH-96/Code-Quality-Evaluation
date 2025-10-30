class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        pizzas.sort()
        n = len(pizzas)
        total_weight = 0
        for i in range(n // 4):
            if (i + 1) % 2 == 1:
                total_weight += pizzas[n - 1 - i]
            else:
                total_weight += pizzas[n - 1 - i]
        return total_weight