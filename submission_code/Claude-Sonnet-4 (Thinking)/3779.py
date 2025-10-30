class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        pizzas.sort(reverse=True)
        n = len(pizzas)
        days = n // 4
        
        total_gain = 0
        left = 0  # Index of the largest remaining pizza
        right = n - 1  # Index of the smallest remaining pizza
        
        for day in range(1, days + 1):
            if day % 2 == 1:  # Odd day
                # Take the largest remaining pizza (contributes to gain)
                gain = pizzas[left]
                left += 1
                
                # Take 3 smallest remaining pizzas (don't affect gain)
                right -= 3
            
            else:  # Even day
                # Take 4 largest remaining pizzas
                # The 2nd largest among them contributes to gain
                gain = pizzas[left + 1]
                left += 4
            
            total_gain += gain
        
        return total_gain