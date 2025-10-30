class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        graph1 = defaultdict(dict)
        for i in range(len(pairs1)):
            start, end = pairs1[i]
            graph1[start][end] = rates1[i]
            graph1[end][start] = 1 / rates1[i]

        graph2 = defaultdict(dict)
        for i in range(len(pairs2)):
            start, end = pairs2[i]
            graph2[start][end] = rates2[i]
            graph2[end][start] = 1 / rates2[i]

        def bellman_ford(graph, start_node):
            dist = defaultdict(lambda: float('-inf'))
            dist[start_node] = 1.0
            for _ in range(len(graph) -1):
                for node in graph:
                    for neighbor, weight in graph[node].items():
                        dist[neighbor] = max(dist[neighbor], dist[node] * weight)
            return dist

        dist1 = bellman_ford(graph1, initialCurrency)
        dist2 = bellman_ford(graph2, initialCurrency)

        max_amount = 0.0
        for currency, amount1 in dist1.items():
            if currency in dist2:
                max_amount = max(max_amount, amount1 * dist2[currency])

        return max_amount

from collections import defaultdict
from typing import List