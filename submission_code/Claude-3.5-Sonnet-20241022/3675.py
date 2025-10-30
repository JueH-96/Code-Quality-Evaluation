class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        # Create adjacency list representation
        n = len(edges) + 1  # number of nodes
        adj = [[] for _ in range(n)]
        
        # Sort edges by weight in descending order
        edges.sort(key=lambda x: -x[2])
        
        # Keep track of degree of each node
        degree = [0] * n
        
        total_weight = 0
        
        # Process edges in descending order of weight
        for u, v, w in edges:
            # If adding this edge doesn't violate the k-degree constraint
            if degree[u] < k and degree[v] < k:
                # Add the edge
                adj[u].append(v)
                adj[v].append(u)
                # Update degrees
                degree[u] += 1
                degree[v] += 1
                # Add weight to total
                total_weight += w
        
        return total_weight