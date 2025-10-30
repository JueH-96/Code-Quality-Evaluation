from typing import List

class Solution:
  def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
    A = 26  # Alphabet size
    
    # Initialize dist table for Floyd-Warshall
    # dist[i][j] = minimum cost to change char i to char j
    # Using float('inf') for initially unknown/large costs.
    dist = [[float('inf')] * A for _ in range(A)]
    
    # Populate initial costs
    for i in range(A):
        # Cost to change char i to itself is 0
        dist[i][i] = 0
        
        # Cost to change char i to char (i+1)%A in one step (moving to next char)
        # This cost is associated with starting at char i.
        char_idx_next = (i + 1) % A
        dist[i][char_idx_next] = min(dist[i][char_idx_next], nextCost[i])
        
        # Cost to change char i to char (i-1+A)%A in one step (moving to previous char)
        # This cost is associated with starting at char i.
        char_idx_prev = (i - 1 + A) % A
        dist[i][char_idx_prev] = min(dist[i][char_idx_prev], previousCost[i])
            
    # Floyd-Warshall algorithm
    # After this, dist[i][j] will store the shortest path cost from char i to char j.
    # Iterating k, then i, then j is the standard order for Floyd-Warshall.
    for k_intermediate_char_idx in range(A):  # Intermediate character index
        for i_source_char_idx in range(A):  # Source character index
            for j_target_char_idx in range(A):  # Destination character index
                # If a path from i to k and from k to j exists (i.e., not infinity)
                if dist[i_source_char_idx][k_intermediate_char_idx] != float('inf') and \
                   dist[k_intermediate_char_idx][j_target_char_idx] != float('inf'):
                    
                    cost_via_k = dist[i_source_char_idx][k_intermediate_char_idx] + \
                                 dist[k_intermediate_char_idx][j_target_char_idx]
                    
                    # If the path through k_intermediate_char_idx is shorter, update dist
                    if cost_via_k < dist[i_source_char_idx][j_target_char_idx]:
                        dist[i_source_char_idx][j_target_char_idx] = cost_via_k
                        
    # Calculate total shift distance by summing costs for each character pair (s[i], t[i])
    total_cost = 0
    for char_idx_in_string in range(len(s)):
        s_char_code = ord(s[char_idx_in_string]) - ord('a')
        t_char_code = ord(t[char_idx_in_string]) - ord('a')
        
        cost_for_char_pair = dist[s_char_code][t_char_code]
        
        # It's guaranteed that any character can be transformed into any other,
        # so cost_for_char_pair will be a finite number.
        
        total_cost += cost_for_char_pair
            
    # All costs are integers, so the sum will be an integer.
    # Python integers handle arbitrary precision, so no overflow for total_cost.
    # int() conversion handles cases where total_cost might be float if intermediate sums were floats,
    # but with integer costs and float('inf'), sums are integers or float('inf').
    # Since paths are guaranteed, total_cost will be a finite integer.
    return int(total_cost)