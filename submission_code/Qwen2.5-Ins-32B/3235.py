from typing import List
import collections
import heapq

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # Create a graph to store the costs of changing characters
        graph = collections.defaultdict(dict)
        for o, c, co in zip(original, changed, cost):
            if c not in graph[o] or graph[o][c] > co:
                graph[o][c] = co
        
        # Function to find the minimum cost to change from 'start' to 'end'
        def dijkstra(start, end):
            if start not in graph:
                return -1
            pq = [(0, start)]
            visited = set()
            while pq:
                cost, node = heapq.heappop(pq)
                if node == end:
                    return cost
                if node in visited:
                    continue
                visited.add(node)
                for neighbor, c in graph[node].items():
                    if neighbor not in visited:
                        heapq.heappush(pq, (cost + c, neighbor))
            return -1
        
        # Precompute the minimum cost for all pairs of characters
        min_cost = {}
        for c1 in set(original + changed):
            for c2 in set(original + changed):
                if c1 != c2:
                    min_cost[(c1, c2)] = dijkstra(c1, c2)
        
        # Calculate the total cost to convert source to target
        total_cost = 0
        for s, t in zip(source, target):
            if s != t:
                if (s, t) not in min_cost or min_cost[(s, t)] == -1:
                    return -1
                total_cost += min_cost[(s, t)]
        
        return total_cost