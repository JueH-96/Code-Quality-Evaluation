from typing import List

class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        
        # To maximize the total weight gain, we use a greedy strategy.
        # The gain depends on the largest (Z) or second largest (Y) pizza
        # in a group of 4. This makes larger pizzas more valuable.
        
        # 1. Sort the pizzas to easily access the largest ones.
        pizzas.sort()
        
        n = len(pizzas)
        if n == 0:
            return 0
        
        # 2. Calculate the total number of days and the counts of odd/even days.
        num_days = n // 4
        # `(num_days + 1) // 2` is equivalent to ceil(num_days / 2) for integers.
        num_odd_days = (num_days + 1) // 2
        num_even_days = num_days // 2
        
        total_gain = 0
        
        # Use a pointer to the end of the sorted array to pick the largest pizzas.
        pizza_idx = n - 1
        
        # 3. First, calculate gains from all odd-numbered days.
        # A 'Z' gain is always >= a 'Y' gain, so we prioritize using the largest
        # pizzas for the Z-gains on odd days.
        for _ in range(num_odd_days):
            # The gain is the largest available pizza.
            total_gain += pizzas[pizza_idx]
            # This pizza is now considered used for a gain slot.
            pizza_idx -= 1
            
        # 4. Next, calculate gains from all even-numbered days.
        # For each even day, we gain Y. This requires consuming two pizzas (Y and Z).
        # To maximize Y, we take the two largest *available* pizzas.
        for _ in range(num_even_days):
            # The largest remaining pizza becomes Z for this group (no gain contribution).
            # It is consumed, so we move the pointer.
            pizza_idx -= 1
            
            # The next largest remaining pizza becomes Y (this is the gain).
            total_gain += pizzas[pizza_idx]
            # This pizza is also consumed.
            pizza_idx -= 1
            
        return total_gain