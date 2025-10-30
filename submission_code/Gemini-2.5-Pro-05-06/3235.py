from typing import List

class Solution:
  def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
    NUM_CHARS = 26
    
    # min_costs[i][j] will store the minimum cost to change character 
    # represented by index i to character represented by index j.
    # Character 'a' corresponds to index 0, 'b' to 1, ..., 'z' to 25.
    # Initialize all costs to infinity.
    min_costs = [[float('inf')] * NUM_CHARS for _ in range(NUM_CHARS)]
    
    # The cost to change a character to itself is 0 (by doing nothing).
    for i in range(NUM_CHARS):
        min_costs[i][i] = 0
        
    # Populate the min_costs matrix with the initial direct transformation costs.
    # original[k] can be changed to changed[k] with cost[k].
    # If multiple direct transformations exist for the same pair (x, y),
    # we take the one with the minimum cost.
    for i in range(len(original)):
        u_char_idx = ord(original[i]) - ord('a')
        v_char_idx = ord(changed[i]) - ord('a')
        current_transformation_cost = cost[i]
        
        min_costs[u_char_idx][v_char_idx] = min(min_costs[u_char_idx][v_char_idx], current_transformation_cost)
            
    # Apply the Floyd-Warshall algorithm to find all-pairs shortest paths.
    # This considers all possible intermediate characters for transformations.
    # k_char_idx: index of the intermediate character (0-25)
    for k_char_idx in range(NUM_CHARS):
        # i_char_idx: index of the source character (0-25)
        for i_char_idx in range(NUM_CHARS):
            # j_char_idx: index of the destination character (0-25)
            for j_char_idx in range(NUM_CHARS):
                # If a path from i_char_idx to k_char_idx exists (cost is not infinity) and
                # a path from k_char_idx to j_char_idx exists (cost is not infinity)
                if min_costs[i_char_idx][k_char_idx] != float('inf') and \
                   min_costs[k_char_idx][j_char_idx] != float('inf'):
                    # Update the cost if the path i -> k -> j is cheaper
                    # than the current known cost for i -> j.
                    min_costs[i_char_idx][j_char_idx] = min(
                        min_costs[i_char_idx][j_char_idx], 
                        min_costs[i_char_idx][k_char_idx] + min_costs[k_char_idx][j_char_idx]
                    )
    
    # Calculate the total cost to convert the source string to the target string.
    total_conversion_cost = 0
    n = len(source)
    
    for i in range(n):
        s_char_idx = ord(source[i]) - ord('a')
        t_char_idx = ord(target[i]) - ord('a')
        
        # If source and target characters at this position are the same,
        # no transformation is needed, so cost is 0.
        # This also correctly handles min_costs[s_char_idx][t_char_idx] being 0.
        if s_char_idx == t_char_idx:
            continue 
        
        # Get the minimum cost to transform s_char to t_char from our precomputed matrix.
        cost_for_this_char_transform = min_costs[s_char_idx][t_char_idx]
        
        # If the cost is infinity, it means this transformation is impossible.
        # Therefore, the entire source string cannot be converted to target.
        if cost_for_this_char_transform == float('inf'):
            return -1
        
        # Add the cost for this character's transformation to the total.
        total_conversion_cost += cost_for_this_char_transform
            
    return total_conversion_cost