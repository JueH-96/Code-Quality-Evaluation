class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        import heapq
        from collections import defaultdict

        n = len(edges) +1
        adj = defaultdict(list)
        for u, v in edges:
            if v %2 ==1:
                adj[u].append((v,1))
            else:
                adj[u].append((v,2))
            if u %2 ==1:
                adj[v].append((u,1))
            else:
                adj[v].append((u,2))
        
        def dijkstra(start):
            INF = float('inf')
            distances = [INF]*n
            distances[start]=0
            heap = [(0,start)]
            while heap:
                d, u = heapq.heappop(heap)
                if d > distances[u]:
                    continue
                for v, w in adj[u]:
                    if distances[v] > d +w:
                        distances[v]=d +w
                        heapq.heappush(heap, (distances[v], v))
            return distances
        
        # First Dijkstra to find one end of the diameter
        dist1 = dijkstra(0)
        far_node = dist1.index(max(dist1))
        
        # Second Dijkstra from far_node to find the other end
        dist2 = dijkstra(far_node)
        farthest = max(dist2)
        far_node2 = dist2.index(farthest)
        
        # Third Dijkstra from far_node2
        dist3 = dijkstra(far_node2)
        
        # For each node, the timeTaken is the max of dist2 and dist3
        times = [max(dist2[i], dist3[i]) for i in range(n)]
        return times