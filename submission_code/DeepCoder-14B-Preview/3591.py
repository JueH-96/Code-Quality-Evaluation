import heapq

class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: list, previousCost: list) -> int:
        # Build the graph
        graph = [[] for _ in range(26)]
        for i in range(26):
            # Next shift
            next_node = (i + 1) % 26
            graph[i].append((next_node, nextCost[i]))
            # Previous shift
            prev_node = (i - 1) % 26
            graph[i].append((prev_node, previousCost[i]))
        
        # Precompute the cost matrix using Dijkstra's algorithm
        cost_matrix = [[0] * 26 for _ in range(26)]
        for source in range(26):
            dist = [float('inf')] * 26
            dist[source] = 0
            heap = []
            heapq.heappush(heap, (0, source))
            
            while heap:
                current_dist, u = heapq.heappop(heap)
                if current_dist > dist[u]:
                    continue
                for v, w in graph[u]:
                    if dist[v] > dist[u] + w:
                        dist[v] = dist[u] + w
                        heapq.heappush(heap, (dist[v], v))
            
            # Update the cost matrix
            for target in range(26):
                cost_matrix[source][target] = dist[target]
        
        # Calculate the total minimal cost
        total = 0
        for sc, tc in zip(s, t):
            i = ord(sc) - ord('a')
            j = ord(tc) - ord('a')
            total += cost_matrix[i][j]
        
        return total