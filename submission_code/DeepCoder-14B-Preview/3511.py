import heapq

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize the adjacency list with the initial roads
        adj = [[] for _ in range(n)]
        for i in range(n - 1):
            adj[i].append(i + 1)
        
        answer = []
        for u, v in queries:
            adj[u].append(v)
            # Dijkstra's algorithm to find the shortest path from 0 to n-1
            dist = [float('inf')] * n
            dist[0] = 0
            heap = []
            heapq.heappush(heap, (0, 0))
            
            while heap:
                current_dist, node = heapq.heappop(heap)
                if node == n - 1:
                    break
                if current_dist > dist[node]:
                    continue
                for neighbor in adj[node]:
                    if dist[neighbor] > current_dist + 1:
                        dist[neighbor] = current_dist + 1
                        heapq.heappush(heap, (dist[neighbor], neighbor))
            answer.append(dist[n - 1])
        return answer