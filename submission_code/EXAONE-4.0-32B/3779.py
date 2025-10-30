class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        pizzas.sort()
        n = len(pizzas)
        k = n // 4
        o = (k + 1) // 2
        e = k // 2
        
        total = sum(pizzas[-o:])
        
        for j in range(e):
            idx = n - o - 2 * (j + 1)
            total += pizzas[idx]
            
        return total