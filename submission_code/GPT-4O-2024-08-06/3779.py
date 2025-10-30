class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        # Sort the pizzas in descending order to maximize the weight gain
        pizzas.sort(reverse=True)
        
        total_weight_gain = 0
        # Iterate over the pizzas in chunks of 4
        for i in range(0, len(pizzas), 4):
            # On odd-numbered days (1-indexed), gain weight of the largest pizza in the group
            if (i // 4) % 2 == 0:
                total_weight_gain += pizzas[i]
            # On even-numbered days, gain weight of the second largest pizza in the group
            else:
                total_weight_gain += pizzas[i + 1]
        
        return total_weight_gain