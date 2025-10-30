import math
from typing import List

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # Define the number of possible characters (a-z)
        NUM_CHARS = 26
        
        # Initialize the 2D array (adjacency matrix) for all-pairs shortest paths.
        # min_cost[i][j] will store the minimum cost to transform the character
        # corresponding to index 'i' (e.g., 'a' for 0, 'b' for 1)
        # to the character corresponding to index 'j'.
        # Initialize with infinity to represent unreachable paths.
        min_cost = [[math.inf] * NUM_CHARS for _ in range(NUM_CHARS)]
        
        # The cost to change a character to itself is 0.
        for i in range(NUM_CHARS):
            min_cost[i][i] = 0
            
        # Populate the `min_cost` matrix with direct transformation costs provided in the input.
        # If there are multiple direct transformations from char_u to char_v,
        # we take the one with the minimum cost.
        for i in range(len(original)):
            # Convert character to its 0-indexed integer representation ('a' -> 0, 'b' -> 1, etc.)
            u = ord(original[i]) - ord('a')
            v = ord(changed[i]) - ord('a')
            w = cost[i]
            
            min_cost[u][v] = min(min_cost[u][v], w)
            
        # Apply the Floyd-Warshall algorithm.
        # This algorithm finds the shortest paths between all pairs of nodes in a graph.
        # Here, nodes are characters, and edge weights are transformation costs.
        # The 'k' loop represents the intermediate character that can be used to
        # form a path from 'i' to 'j'.
        for k in range(NUM_CHARS):  # Intermediate character (pivot)
            for i in range(NUM_CHARS):  # Starting character
                for j in range(NUM_CHARS):  # Ending character
                    # If there's a path from 'i' to 'k' AND a path from 'k' to 'j'
                    # (i.e., neither path segment has infinite cost),
                    # then we can potentially update the cost from 'i' to 'j'
                    # by going through 'k'.
                    if min_cost[i][k] != math.inf and min_cost[k][j] != math.inf:
                        min_cost[i][j] = min(min_cost[i][j], min_cost[i][k] + min_cost[k][j])
                        
        # After Floyd-Warshall, min_cost[u][v] contains the minimum cost to transform
        # character 'u' to character 'v' using any number of intermediate steps.
        
        # Calculate the total minimum cost to convert the source string to the target string.
        total_min_cost = 0
        for i in range(len(source)):
            s_char_idx = ord(source[i]) - ord('a')
            t_char_idx = ord(target[i]) - ord('a')
            
            # If the characters are already the same, no operation is needed for this position.
            if s_char_idx == t_char_idx:
                continue
            
            # Get the minimum cost to transform source[i] to target[i].
            cost_for_char_pair = min_cost[s_char_idx][t_char_idx]
            
            # If the cost is still infinity, it means it's impossible to transform
            # source[i] to target[i]. In this case, the total conversion is impossible.
            if cost_for_char_pair == math.inf:
                return -1
            
            # Add the cost for this character pair to the total.
            total_min_cost += cost_for_char_pair
            
        # Return the accumulated total minimum cost.
        return total_min_cost