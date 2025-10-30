class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        n = len(pizzas)
        pizzas.sort()
        
        total_weight = 0
        for i in range(0, n, 4):
            w, x, y, z = pizzas[i], pizzas[i+1], pizzas[i+2], pizzas[i+3]
            total_weight += z
            if i % 2 == 0:
                total_weight += y
            else:
                total_weight += z
        
        return total_weight