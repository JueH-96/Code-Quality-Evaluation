from typing import List

class Solution:
  def minCosts(self, cost: List[int]) -> List[int]:
    n = len(cost)
    answer = [0] * n  # Pre-allocate the answer array.
    
    # This variable will keep track of the minimum cost found in cost[0...i]
    # as we iterate i. According to the logic derived, 
    # answer[i] = min(cost[0], cost[1], ..., cost[i]).
    current_min_prefix_cost = float('inf')
    
    for i in range(n):
      # Update the minimum cost encountered so far by considering cost[i].
      # This ensures current_min_prefix_cost = min(cost[0]...cost[i]).
      current_min_prefix_cost = min(current_min_prefix_cost, cost[i])
      
      # The minimum cost to reach position i is this running minimum.
      answer[i] = current_min_prefix_cost
            
    return answer