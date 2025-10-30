import heapq

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # Build the graph
        adj = [[] for _ in range(26)]
        for j in range(len(original)):
            o = original[j]
            c = changed[j]
            z = cost[j]
            o_idx = ord(o) - ord('a')
            c_idx = ord(c) - ord('a')
            adj[o_idx].append((c_idx, z))
        
        # Precompute all-pairs shortest paths using Dijkstra's algorithm
        INF = float('inf')
        dist = [[INF] * 26 for _ in range(26)]
        for x in range(26):
            # Priority queue: (distance, node)
            pq = []
            dist[x][x] = 0
            heapq.heappush(pq, (0, x))
            visited = [False] * 26
            while pq:
                current_dist, u = heapq.heappop(pq)
                if visited[u]:
                    continue
                visited[u] = True
                for v, w in adj[u]:
                    if dist[x][v] > current_dist + w:
                        dist[x][v] = current_dist + w
                        heapq.heappush(pq, (dist[x][v], v))
        
        # Calculate the total cost
        total_cost = 0
        for i in range(len(source)):
            s = source[i]
            t = target[i]
            if s == t:
                continue
            s_idx = ord(s) - ord('a')
            t_idx = ord(t) - ord('a')
            if dist[s_idx][t_idx] == INF:
                return -1
            total_cost += dist[s_idx][t_idx]
        
        return total_cost