import collections
import heapq

class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        n = len(edges) + 1
        total_original = sum(w for u, v, w in edges)
        deg = [0] * n
        graph = [[] for _ in range(n)]
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))
            deg[u] += 1
            deg[v] += 1
        
        heaps = [[] for _ in range(n)]
        for i in range(n):
            for neighbor, w in graph[i]:
                heapq.heappush(heaps[i], (w, neighbor))
        
        q = collections.deque()
        for i in range(n):
            if deg[i] > k:
                q.append(i)
        
        removed_edges = set()
        total_removed = 0
        
        while q:
            u = q.popleft()
            if deg[u] <= k:
                continue
            valid_edge_found = False
            while heaps[u] and not valid_edge_found:
                w, v = heapq.heappop(heaps[u])
                edge_id = (min(u, v), max(u, v))
                if edge_id in removed_edges:
                    continue
                removed_edges.add(edge_id)
                total_removed += w
                deg[u] -= 1
                deg[v] -= 1
                if deg[u] > k:
                    q.append(u)
                if deg[v] > k:
                    q.append(v)
                valid_edge_found = True
        
        return total_original - total_removed