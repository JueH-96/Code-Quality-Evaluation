class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        n = 0
        for edge in edges:
            n = max(n, edge[0], edge[1])
        n += 1
        adj = [[] for _ in range(n)]
        degrees = [0] * n
        node_edges = [[] for _ in range(n)]
        for i in range(len(edges)):
            u, v, w = edges[i]
            adj[u].append((v, w, i))
            adj[v].append((u, w, i))
            degrees[u] += 1
            degrees[v] += 1
            node_edges[u].append((v, w, i))
            node_edges[v].append((u, w, i))
            
        removed_edges_indices = set()
        
        for u in range(n):
            if degrees[u] > k:
                current_edges = sorted(node_edges[u], key=lambda x: x[1])
                num_remove = degrees[u] - k
                removed_count = 0
                for v_node, weight, edge_index in current_edges:
                    if edge_index not in removed_edges_indices:
                        removed_edges_indices.add(edge_index)
                        degrees[u] -= 1
                        degrees[v_node] -= 1
                        removed_count += 1
                        if removed_count == num_remove:
                            break
                            
        total_weight = 0
        for i in range(len(edges)):
            if i not in removed_edges_indices:
                total_weight += edges[i][2]
                
        return total_weight