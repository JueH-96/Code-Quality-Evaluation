class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        pizzas_sorted = sorted(pizzas, reverse=True)
        n = len(pizzas)
        k = n // 4
        m = (k + 1) // 2  # number of odd days
        
        total = sum(pizzas_sorted[:m])
        
        remaining_start = m
        remaining_end = 4 * k - 3 * m
        remaining = pizzas_sorted[remaining_start:remaining_end]
        
        for i in range(0, len(remaining), 4):
            total += remaining[i + 1]
        
        return total