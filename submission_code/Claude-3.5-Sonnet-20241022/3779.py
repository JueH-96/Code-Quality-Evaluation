class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        # Sort pizzas in ascending order
        pizzas.sort()
        n = len(pizzas)
        total_days = n // 4
        total_weight = 0
        
        # For each day, we need to pick 4 pizzas
        # We'll process from largest to smallest for odd days (to maximize Z)
        # and then process remaining for even days (to maximize Y)
        
        # First, handle odd days (getting Z values)
        odd_day_indices = []
        for i in range(total_days):
            if i % 2 == 0:  # odd day (1-indexed)
                # Take the largest available pizza as Z
                total_weight += pizzas[n - 1 - i * 4]
                # Mark the indices we'll use for this day
                odd_day_indices.extend([
                    n - 1 - i * 4,      # Z
                    n - 2 - i * 4,      # Y
                    n - 3 - i * 4,      # X
                    n - 4 - i * 4       # W
                ])
        
        # Now handle even days with remaining pizzas
        remaining_pizzas = []
        used = set(odd_day_indices)
        for i in range(n):
            if i not in used:
                remaining_pizzas.append(pizzas[i])
        
        # Process even days
        for i in range(0, len(remaining_pizzas), 4):
            # On even days, we gain Y (second largest of the group)
            total_weight += remaining_pizzas[i + 2]
            
        return total_weight