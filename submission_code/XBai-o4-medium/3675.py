class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        # Determine the number of nodes, which is edges length + 1
        n = len(edges) + 1
        
        # Initialize the degrees array to track how many edges each node has been kept with
        degrees = [0] * n
        
        # Sort the edges in descending order based on their weights
        edges.sort(key=lambda x: -x[2])
        
        total = 0
        
        for u, v, w in edges:
            if degrees[u] < k and degrees[v] < k:
                total += w
                degrees[u] += 1
                degrees[v] += 1
        
        return total