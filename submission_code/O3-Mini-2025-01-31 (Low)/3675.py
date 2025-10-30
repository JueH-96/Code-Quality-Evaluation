from typing import List

class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        # Our approach:
        # We need to choose a subset of the tree's edges such that the degree of any node is at most k,
        # and the total sum of the weights is maximized.
        # Because the original graph is a tree, we can simply decide edge-by-edge whether to include 
        # it, under the degree constraints.
        # For a maximum solution, we should include the higher weight edges first.
        #
        # We maintain an available degree for each node, initially k for each.
        # Then, we sort the edges in descending order by weight.
        # For each edge, if both endpoints have at least one available degree, we include it 
        # (adding its weight to the answer and decrementing the available degree of both endpoints).
        #
        # This greedy algorithm is correct in a tree since each edge is independent in the sense 
        # that including one edge only blocks additional edges incident to its endpoints.
        # This approach is similar to finding a maximum spanning forest (here the forest is from the original tree)
        # under degree constraints.
        
        # Initialize available degrees
        n = 0
        for u, v, _ in edges:
            n = max(n, u, v)
        n += 1  # since nodes are 0-indexed
        available = [k] * n
        
        # Sort edges in descending order by weight
        edges.sort(key=lambda x: x[2], reverse=True)
        
        result = 0
        # Iterate over edges; greedy selection if both endpoints can accommodate another edge.
        for u, v, w in edges:
            if available[u] > 0 and available[v] > 0:
                result += w
                available[u] -= 1
                available[v] -= 1
        
        return result