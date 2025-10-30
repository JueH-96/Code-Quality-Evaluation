from typing import List

class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        # Sort the pizza weights
        pizzas.sort()
        n = len(pizzas)
        
        # Two pointers: l for the smallest, r for the largest
        l, r = 0, n - 1
        gain = 0
        day = 1
        
        # We will run exactly n/4 days, but we'll break when pointers cross
        while l <= r:
            if day % 2 == 1:
                # Odd day: gain the weight of the largest pizza Z
                gain += pizzas[r]
                # Remove Z and the three smallest pizzas (W, X, Y)
                r -= 1
                l += 3
            else:
                # Even day: gain the weight of the second-largest pizza Y
                # We pick the largest (Z) and second-largest (Y) from the top,
                # and two smallest from the bottom
                gain += pizzas[r - 1]
                r -= 2  # remove Z and Y
                l += 2  # remove two smallest (W, X)
            
            day += 1
        
        return gain