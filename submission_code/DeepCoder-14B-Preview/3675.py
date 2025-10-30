class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        n = len(edges) + 1
        node_edges = [[] for _ in range(n)]
        for u, v, w in edges:
            node_edges[u].append((v, w))
            node_edges[v].append((u, w))
        
        sum_total = sum(w for u, v, w in edges)
        
        removed = set()
        sum_removed = 0
        
        for u in range(n):
            edges_list = node_edges[u]
            if len(edges_list) > k:
                sorted_edges = sorted(edges_list, key=lambda x: x[1])
                num_remove = len(sorted_edges) - k
                for i in range(num_remove):
                    v, w = sorted_edges[i]
                    key = tuple(sorted((u, v)))
                    if key not in removed:
                        removed.add(key)
                        sum_removed += w
        
        return sum_total - sum_removed