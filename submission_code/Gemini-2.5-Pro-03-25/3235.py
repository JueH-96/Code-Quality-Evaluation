import math
from typing import List 

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        """
        Calculates the minimum cost to convert string source to string target using character transformations.

        The problem can be viewed as finding the shortest paths in a directed graph where nodes represent the 26 lowercase English letters.
        An edge from character `x` to character `y` with weight `z` exists if a transformation from `x` to `y` with cost `z` is allowed.
        The minimum cost to transform `source[i]` to `target[i]` for any index `i` is equivalent to the shortest path cost
        from the node corresponding to `source[i]` to the node corresponding to `target[i]`.
        The total minimum cost to convert the entire `source` string to `target` is the sum of these minimum costs for all indices `i`.

        We use the Floyd-Warshall algorithm to compute the shortest paths (minimum costs) between all pairs of characters.
        This algorithm is suitable here because the number of nodes (characters) is small (26).

        Args:
            source: The starting string.
            target: The target string. Both source and target have the same length and consist of lowercase English letters.
            original: A list of characters that can be changed. `original[k]` is the character to change from.
            changed: A list of characters corresponding to the result of changing `original[k]`. `changed[k]` is the character to change to.
            cost: A list of costs corresponding to the change `original[k]` -> `changed[k]`. `cost[k]` is the cost.

        Returns:
            The minimum total cost to convert `source` to `target`. If the conversion is impossible for any character, returns -1.
        """
        
        num_chars = 26 # There are 26 lowercase English letters ('a' through 'z')
        
        # Initialize a 2D list (adjacency matrix representation of the graph)
        # to store minimum costs between characters.
        # min_cost_matrix[i][j] will hold the minimum cost to change character 
        # represented by index i (0 for 'a', 1 for 'b', ..., 25 for 'z') 
        # to character represented by index j.
        # Initialize costs to infinity, representing that paths are initially unknown or impossible.
        # Use float('inf') for infinity which works well with comparisons and additions.
        min_cost_matrix = [[float('inf')] * num_chars for _ in range(num_chars)]
        
        # The cost to change a character to itself is always 0.
        # Initialize diagonal elements of the matrix to 0.
        for i in range(num_chars):
            min_cost_matrix[i][i] = 0
            
        # Populate the matrix with the costs of the given direct transformations.
        # The input might specify multiple costs for the same transformation pair (e.g., a -> b with cost 5, and a -> b with cost 3).
        # We are interested in the minimum cost for any direct transformation, so we take the minimum.
        for k in range(len(original)):
            # Convert characters ('a' to 'z') to corresponding indices (0 to 25).
            # ord(char) gives the ASCII value. Subtracting ord('a') maps 'a' to 0, 'b' to 1, etc.
            u = ord(original[k]) - ord('a')
            v = ord(changed[k]) - ord('a')
            current_cost = cost[k]
            
            # Update the minimum cost for the direct transformation u -> v if the current one is cheaper.
            min_cost_matrix[u][v] = min(min_cost_matrix[u][v], current_cost)
            
        # Apply the Floyd-Warshall algorithm to compute all-pairs shortest paths.
        # This algorithm finds the minimum cost path between all pairs of characters,
        # considering paths that may involve one or more intermediate characters.
        # For example, if we can change 'a' to 'c' (cost C1) and 'c' to 'b' (cost C2), Floyd-Warshall
        # will determine if the path a -> c -> b (total cost C1 + C2) is the minimum cost path from 'a' to 'b'.
        for k in range(num_chars):  # k represents the intermediate character/node index
            for i in range(num_chars):  # i represents the source character/node index
                # Optimization: If the path from i to k is impossible (cost is infinity),
                # then any path from i passing through k will also be impossible or not yield a better finite cost. Skip.
                if min_cost_matrix[i][k] == float('inf'):
                    continue
                for j in range(num_chars):  # j represents the destination character/node index
                    # Optimization: If the path from k to j is impossible (cost is infinity),
                    # then the path i -> k -> j is impossible. Skip.
                    if min_cost_matrix[k][j] == float('inf'):
                        continue
                    
                    # Calculate the cost of the path from i to j through k.
                    path_cost_via_k = min_cost_matrix[i][k] + min_cost_matrix[k][j]
                    
                    # If this path (i -> k -> j) is cheaper than the currently known shortest path cost from i to j,
                    # update the minimum cost for path i -> j.
                    if path_cost_via_k < min_cost_matrix[i][j]:
                        min_cost_matrix[i][j] = path_cost_via_k

        # Calculate the total minimum cost required to transform the source string to the target string.
        total_cost = 0
        n = len(source) # Length of the source and target strings (they have equal length).
        
        # Iterate through each character position in the strings from index 0 to n-1.
        for i in range(n):
            # If the characters at the current position i are already the same in both source and target strings,
            # no transformation is needed for this position, so the cost is 0. Continue to the next position.
            if source[i] == target[i]:
                continue
            
            # Convert the source and target characters at position i to their corresponding indices (0-25).
            source_idx = ord(source[i]) - ord('a')
            target_idx = ord(target[i]) - ord('a')
            
            # Look up the minimum cost required to transform the character source[i] to target[i].
            # This cost represents the shortest path cost found by Floyd-Warshall and stored in min_cost_matrix.
            cost_for_char = min_cost_matrix[source_idx][target_idx]
            
            # Check if the transformation is possible. If the cost is infinity (float('inf')), it means
            # there is no sequence of allowed transformations to change source[i] to target[i].
            # In this scenario, the overall transformation from the source string to the target string is impossible. Return -1.
            if cost_for_char == float('inf'):
                return -1
            
            # If the transformation is possible (cost is finite), add this minimum cost to the total cost accumulator.
            total_cost += cost_for_char
            
        # After iterating through all positions, if we haven't returned -1, it means all necessary
        # character transformations are possible. The variable total_cost now holds the sum
        # of minimum costs for all positions. Return this total minimum cost.
        # Since all input costs are integers, the sum of costs along any path is an integer.
        # Thus, total_cost will be an integer. We return it as such.
        return int(total_cost) # Cast to int for explicit type guarantee, though it should already be effectively integer.