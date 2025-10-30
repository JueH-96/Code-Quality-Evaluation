from collections import defaultdict
import heapq

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # Create a graph to represent the possible transformations
        graph = defaultdict(dict)
        for o, c, co in zip(original, changed, cost):
            if c in graph[o]:
                if co < graph[o][c]:
                    graph[o][c] = co
            else:
                graph[o][c] = co
        
        # Precompute the shortest path between all pairs of characters using Dijkstra's algorithm
        # Initialize the distance matrix
        chars = set(original + changed)
        distance = {char: {} for char in chars}
        for char in chars:
            distance[char][char] = 0
            heap = [(0, char)]
            while heap:
                current_dist, u = heapq.heappop(heap)
                if u in distance[char] and current_dist > distance[char][u]:
                    continue
                for v, w in graph[u].items():
                    if v not in distance[char] or distance[char][v] > current_dist + w:
                        distance[char][v] = current_dist + w
                        heapq.heappush(heap, (distance[char][v], v))
        
        total_cost = 0
        for s_char, t_char in zip(source, target):
            if s_char == t_char:
                continue
            if s_char not in distance or t_char not in distance[s_char]:
                return -1
            total_cost += distance[s_char][t_char]
        return total_cost