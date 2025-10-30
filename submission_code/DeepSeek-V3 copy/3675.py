from collections import defaultdict

class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        # Create a graph to store the edges for each node
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))
        
        # Initialize the degree of each node
        degree = defaultdict(int)
        for u, v, _ in edges:
            degree[u] += 1
            degree[v] += 1
        
        # Sort edges by weight in descending order
        edges_sorted = sorted(edges, key=lambda x: -x[2])
        
        # Initialize the sum of weights
        total_weight = 0
        
        # Iterate through the sorted edges
        for u, v, w in edges_sorted:
            if degree[u] > k and degree[v] > k:
                # If both nodes have degree greater than k, we need to remove this edge
                # Decrease the degree of both nodes
                degree[u] -= 1
                degree[v] -= 1
            else:
                # Otherwise, add the weight to the total
                total_weight += w
        
        return total_weight