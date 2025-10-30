from typing import List

class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        # Sort the pizzas in descending order to maximize the weight gain
        pizzas.sort(reverse=True)
        
        total_weight = 0
        # Iterate over the pizzas in chunks of 4
        for i in range(0, len(pizzas), 4):
            # On odd days (1-indexed), we gain the weight of the heaviest pizza (Z)
            total_weight += pizzas[i]  # Z is the first pizza in the sorted group
            # On even days (1-indexed), we gain the weight of the second heaviest pizza (Y)
            if i + 1 < len(pizzas):
                total_weight += pizzas[i + 1]  # Y is the second pizza in the sorted group
        
        return total_weight