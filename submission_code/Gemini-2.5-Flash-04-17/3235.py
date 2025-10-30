from typing import List
import math

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # Number of lowercase English letters
        num_chars = 26
        
        # Initialize the distance matrix for all pairs of characters
        # dist[i][j] will store the minimum cost to change character i to character j
        # Initialize with infinity, except for dist[i][i] which is 0
        dist = [[float('inf')] * num_chars for _ in range(num_chars)]
        for i in range(num_chars):
            dist[i][i] = 0

        # Populate the initial distances based on the given transformations
        # If there are multiple ways to change original[k] to changed[k], take the minimum cost
        for k in range(len(original)):
            u = ord(original[k]) - ord('a')
            v = ord(changed[k]) - ord('a')
            dist[u][v] = min(dist[u][v], cost[k])

        # Apply Floyd-Warshall algorithm to find the all-pairs shortest paths
        # This considers all possible intermediate characters for transformations
        for k in range(num_chars): # Intermediate character (0 to 25)
            for i in range(num_chars): # Starting character (0 to 25)
                for j in range(num_chars): # Ending character (0 to 25)
                    # If a path exists from i to k and from k to j (i.e., costs are not infinity)
                    # Note: adding infinity to any finite number results in infinity.
                    # We only update if both dist[i][k] and dist[k][j] are finite.
                    if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                        # Update the minimum cost from i to j by considering the path through k
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

        # Calculate the total minimum cost to transform source to target
        total_cost = 0
        n = len(source)
        for i in range(n):
            # If the characters at the current index are different, we need a transformation
            if source[i] != target[i]:
                u = ord(source[i]) - ord('a')
                v = ord(target[i]) - ord('a')
                
                # If the minimum cost to transform source[i] to target[i] is still infinity,
                # it means the transformation is impossible
                if dist[u][v] == float('inf'):
                    return -1
                    
                # Add the minimum transformation cost to the total cost
                # Python handles large integers automatically, so overflow is not a concern here.
                total_cost += dist[u][v]

        # Return the total minimum cost
        return total_cost