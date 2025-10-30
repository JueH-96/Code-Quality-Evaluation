from typing import List

class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        pizzas.sort()
        pizzas = pizzas[::-1]  # Reverse to get descending order
        m = len(pizzas) // 4
        total = 0
        for i in range(m):
            total += pizzas[2 * i]
        return total