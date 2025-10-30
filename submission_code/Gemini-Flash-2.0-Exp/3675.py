class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        n = 0
        for edge in edges:
            n = max(n, edge[0], edge[1])
        n += 1
        adj = [[] for _ in range(n)]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        
        total_weight = 0
        removed_edges = set()
        
        for u in range(n):
            while len(adj[u]) > k:
                
                min_weight = float('inf')
                min_v = -1
                
                for i in range(len(adj[u])):
                    v, w = adj[u][i]
                    if w < min_weight:
                        min_weight = w
                        min_v = v
                        min_index_u = i
                
                adj[u].pop(min_index_u)
                
                for i in range(len(adj[min_v])):
                    if adj[min_v][i][0] == u:
                        adj[min_v].pop(i)
                        break
        
        
        for u, v, w in edges:
            total_weight += w
        
        remaining_weight = 0
        
        used_edges = set()
        
        for u in range(n):
            for v, w in adj[u]:
                if (u,v) not in used_edges and (v,u) not in used_edges:
                    remaining_weight += w
                    used_edges.add((u,v))
                    used_edges.add((v,u))
        
        return remaining_weight // 2