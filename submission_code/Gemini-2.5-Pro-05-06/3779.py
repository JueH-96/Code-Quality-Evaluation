from typing import List

class Solution:
  def maxWeight(self, pizzas: List[int]) -> int:
    n = len(pizzas)
    # Constraints: 4 <= n. So n=0 check is not strictly needed.
    # n is guaranteed to be a multiple of 4.
    
    pizzas.sort()
    
    total_weight_gained = 0
    left = 0
    right = n - 1
    
    num_days = n // 4
    
    # Iterate for each day. Day numbers are 1-indexed for problem's odd/even rule.
    for day_num in range(1, num_days + 1):
      if day_num % 2 == 1: # Odd-numbered day (1st, 3rd, 5th, ... day)
        # We eat 4 pizzas: W, X, Y, Z, sorted by weight W <= X <= Y <= Z.
        # On an odd day, we gain weight Z (the largest).
        # To maximize Z, we choose pizzas[right] (current largest available) as Z.
        # The other 3 pizzas (W, X, Y) are chosen from the smallest available to preserve larger pizzas
        # for future days. These are pizzas[left], pizzas[left+1], pizzas[left+2].
        # The condition W <= X <= Y <= Z is met because pizzas is sorted and indices used are distinct
        # such that elements from left are smaller than elements from right. Specifically,
        # pizzas[left] <= pizzas[left+1] <= pizzas[left+2] <= pizzas[right]
        # because left+2 < right (since at least 4 pizzas are always available for selection,
        # making the indices distinct and ordered).
        
        total_weight_gained += pizzas[right]
        
        # Consume pizzas:
        right -= 1 # Z is consumed
        left += 3  # W, X, Y are consumed
      else: # Even-numbered day (2nd, 4th, 6th, ... day)
        # We eat 4 pizzas: W, X, Y, Z, sorted by weight W <= X <= Y <= Z.
        # On an even day, we gain weight Y (the second largest).
        # To maximize Y, we choose pizzas[right-1] as Y and pizzas[right] as Z.
        # (These are the two largest available pizzas).
        # The other 2 pizzas (W, X) are chosen from the smallest available.
        # These are pizzas[left], pizzas[left+1].
        # The condition W <= X <= Y <= Z is met because pizzas is sorted and indices used are distinct.
        # Specifically, pizzas[left] <= pizzas[left+1] <= pizzas[right-1] <= pizzas[right]
        # because left+1 < right-1 (since at least 4 pizzas are always available for selection).

        total_weight_gained += pizzas[right-1]
        
        # Consume pizzas:
        right -= 2 # Y and Z are consumed
        left += 2  # W and X are consumed
        
    return total_weight_gained