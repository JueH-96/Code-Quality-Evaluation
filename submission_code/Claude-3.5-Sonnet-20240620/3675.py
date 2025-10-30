from collections import defaultdict
import heapq

class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        n = len(edges) + 1
        graph = defaultdict(list)
        
        # Build the graph
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))
        
        def dfs(node, parent):
            heap = []
            total_weight = 0
            
            for child, weight in graph[node]:
                if child != parent:
                    child_weight = dfs(child, node)
                    total_weight += child_weight
                    heapq.heappush(heap, (-child_weight, weight))
            
            # Keep at most k heaviest edges
            while len(heap) > k:
                child_weight, edge_weight = heapq.heappop(heap)
                total_weight += child_weight + edge_weight
            
            return total_weight
        
        return dfs(0, -1)