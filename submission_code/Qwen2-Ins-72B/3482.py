from collections import defaultdict
from heapq import heappush, heappop

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        # Create a graph where nodes are prefixes of the target string
        # and edges are transitions from one prefix to another by appending a word.
        graph = defaultdict(list)
        for word, cost in zip(words, costs):
            for i in range(len(target) - len(word) + 1):
                prefix = target[:i]
                next_prefix = target[:i+len(word)]
                graph[prefix].append((next_prefix, cost))
        
        # Use Dijkstra's algorithm to find the shortest path from the empty string
        # to the target string.
        heap = [(0, "")]
        visited = set()
        while heap:
            cost, prefix = heappop(heap)
            if prefix == target:
                return cost
            if prefix in visited:
                continue
            visited.add(prefix)
            for next_prefix, edge_cost in graph[prefix]:
                if next_prefix not in visited:
                    heappush(heap, (cost + edge_cost, next_prefix))
        
        # If the target string cannot be reached, return -1.
        return -1