class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        # Sort the pizzas array to easily pick the largest weights
        pizzas.sort()
        
        n = len(pizzas)
        total_weight = 0
        day_count = 0
        
        # Start from the end of the sorted list and pick groups of 4
        for i in range(n - 1, -1, -4):
            if day_count % 2 == 0:  # Odd-numbered day (1-indexed)
                total_weight += pizzas[i]  # Z value (largest in the current group of 4)
            else:  # Even-numbered day
                total_weight += pizzas[i - 1]  # Y value (second largest in the current group)
            day_count += 1
        
        return total_weight