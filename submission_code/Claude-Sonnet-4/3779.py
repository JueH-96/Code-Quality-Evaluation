class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        pizzas.sort(reverse=True)
        n = len(pizzas)
        total_days = n // 4
        
        total_weight = 0
        
        # We need to assign pizzas optimally
        # The key insight: we have total_days positions that will contribute to our score
        # Half of these are odd days (Z positions), half are even days (Y positions)
        
        odd_days = (total_days + 1) // 2  # Number of odd days
        even_days = total_days // 2       # Number of even days
        
        # Assign the heaviest pizzas to the positions that count
        for i in range(odd_days):
            total_weight += pizzas[i]  # These go to Z positions (odd days)
        
        for i in range(even_days):
            total_weight += pizzas[odd_days + i]  # These go to Y positions (even days)
        
        return total_weight