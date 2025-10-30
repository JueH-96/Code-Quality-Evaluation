class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        pizzas.sort()
        n = len(pizzas)
        k = n // 4
        k1 = (k + 1) // 2
        k2 = k - k1
        total_odd = sum(pizzas[n - k1 : n])
        total_even = sum(pizzas[n - k1 - 2 * k2 : n - k1 : 2])
        return total_odd + total_even