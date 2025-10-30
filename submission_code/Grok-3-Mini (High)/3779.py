from typing import List

class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        n = len(pizzas)
        d = n // 4
        k = (d + 1) // 2  # Number of odd days
        m = d // 2       # Number of even days
        pizzas_sorted = sorted(pizzas, reverse=True)
        sum_gain = sum(pizzas_sorted[:k]) + sum(pizzas_sorted[i] for i in range(k + 1, k + 2 * m, 2))
        return sum_gain