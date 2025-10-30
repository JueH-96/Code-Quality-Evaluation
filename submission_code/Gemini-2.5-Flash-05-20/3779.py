from typing import List

class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        # Sort the pizzas in ascending order.
        # This allows us to easily access the smallest pizzas from the left
        # and the largest pizzas from the right.
        pizzas.sort()

        n = len(pizzas)
        total_weight_gained = 0
        
        # Initialize two pointers:
        # `left` points to the smallest available pizza.
        # `right` points to the largest available pizza.
        left = 0
        right = n - 1
        
        # Keep track of the day number (1-indexed).
        day_count = 1

        # We will form n/4 groups of 4 pizzas.
        # The loop runs until all pizzas are consumed.
        while left <= right:
            if day_count % 2 == 1:  # Odd-numbered day
                # On an odd day, we gain Z (the largest weight in the group).
                # To maximize this, we take the largest available pizza as Z.
                # For the other three pizzas (W, X, Y), we take the smallest available
                # to preserve larger pizzas for future turns.
                # The group formed will be [pizzas[left], pizzas[left+1], pizzas[left+2], pizzas[right]].
                # We gain pizzas[right].
                total_weight_gained += pizzas[right]
                
                # Consume the 3 smallest pizzas from the left and 1 largest from the right.
                left += 3
                right -= 1
            else:  # Even-numbered day
                # On an even day, we gain Y (the second largest weight in the group).
                # To maximize this, we take the second largest available pizza as Y.
                # The largest available pizza will naturally be Z.
                # For the remaining two pizzas (W, X), we take the smallest available.
                # The group formed will be [pizzas[left], pizzas[left+1], pizzas[right-1], pizzas[right]].
                # We gain pizzas[right-1].
                total_weight_gained += pizzas[right - 1]
                
                # Consume the 2 smallest pizzas from the left and 2 largest from the right.
                left += 2
                right -= 2
            
            day_count += 1
            
        return total_weight_gained