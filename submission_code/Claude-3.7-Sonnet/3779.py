class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        pizzas.sort(reverse=True)  # Sort in descending order
        days = len(pizzas) // 4
        pairs = days // 2
        extra_odd_day = days % 2 == 1
        
        total_gained = 0
        
        for i in range(pairs):
            # For each pair of (odd, even) days
            total_gained += pizzas[i * 4]       # Gain the largest for the odd day
            total_gained += pizzas[i * 4 + 2]   # Gain the third-largest for the even day
        
        if extra_odd_day:
            # For the extra odd day
            total_gained += pizzas[pairs * 4]  # Gain the largest for the odd day
        
        return total_gained