class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        n = len(pizzas)
        pizzas.sort(reverse=True)
        
        # Number of days
        num_days = n // 4
        
        # Number of odd days and even days
        odd_days = (num_days + 1) // 2
        even_days = num_days // 2
        
        total_weight = 0
        
        # For odd days, we take the largest pizzas as Z
        for i in range(odd_days):
            total_weight += pizzas[i]
        
        # For even days, we take the next largest pizzas as Y
        for i in range(odd_days, odd_days + even_days):
            total_weight += pizzas[i]
        
        return total_weight