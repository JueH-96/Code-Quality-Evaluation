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
        
        # Precompute the shortest path between every pair of characters using Dijkstra's algorithm
        # Initialize the distance matrix
        chars = set(original + changed)
        distance = {char: {} for char in chars}
        for char in chars:
            heap = []
            heapq.heappush(heap, (0, char))
            distance[char][char] = 0
            while heap:
                current_cost, current_char = heapq.heappop(heap)
                if current_char not in graph:
                    continue
                for neighbor, weight in graph[current_char].items():
                    if neighbor not in distance[char] or distance[char][neighbor] > current_cost + weight:
                        distance[char][neighbor] = current_cost + weight
                        heapq.heappush(heap, (current_cost + weight, neighbor))
        
        total_cost = 0
        for s_char, t_char in zip(source, target):
            if s_char == t_char:
                continue
            if s_char not in distance or t_char not in distance[s_char]:
                return -1
            total_cost += distance[s_char][t_char]
        
        return total_cost