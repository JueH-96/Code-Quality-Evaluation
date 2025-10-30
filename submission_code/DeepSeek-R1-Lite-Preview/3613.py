import math
from typing import List
import heapq

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        # Function to build the graph
        def build_graph(pairs, rates):
            graph = {}
            for (a, b), rate in zip(pairs, rates):
                if a not in graph:
                    graph[a] = []
                if b not in graph:
                    graph[b] = []
                # Edge from a to b with weight -log(rate)
                graph[a].append((b, -math.log(rate)))
                # Edge from b to a with weight log(rate)
                graph[b].append((a, math.log(rate)))
            return graph
        
        # Dijkstra's algorithm to find shortest paths from start_node
        def dijkstra(graph, start_node):
            distances = {}
            for node in graph:
                distances[node] = float('inf')
            distances[start_node] = 0
            heap = [(0, start_node)]
            while heap:
                current_dist, u = heapq.heappop(heap)
                if current_dist > distances[u]:
                    continue
                for v, weight in graph[u]:
                    if distances[v] > distances[u] + weight:
                        distances[v] = distances[u] + weight
                        heapq.heappush(heap, (distances[v], v))
            # Convert distances to multipliers
            multipliers = {}
            for node in graph:
                if distances[node] != float('inf'):
                    multiplier = math.exp(-distances[node])
                    multipliers[node] = multiplier
                else:
                    multipliers[node] = 0.0
            return multipliers
        
        # Collect all unique currencies
        currencies = set()
        for pair in pairs1:
            currencies.add(pair[0])
            currencies.add(pair[1])
        for pair in pairs2:
            currencies.add(pair[0])
            currencies.add(pair[1])
        
        # Build day 1 graph
        graph_day1 = build_graph(pairs1, rates1)
        # Build day 2 graph
        graph_day2 = build_graph(pairs2, rates2)
        
        # Run Dijkstra's for day 1
        multiplier_day1 = dijkstra(graph_day1, initialCurrency)
        # Run Dijkstra's for day 2 from initialCurrency
        multiplier_day2_initial_to_currency = dijkstra(graph_day2, initialCurrency)
        
        # Compute total amount for each currency
        max_amount = 0.0
        for currency in currencies:
            # multiplier_day2[currency] = 1 / multiplier_day2_initial_to_currency[currency]
            if multiplier_day2_initial_to_currency[currency] == 0:
                continue  # No path back to initialCurrency
            multiplier_day2 = 1 / multiplier_day2_initial_to_currency[currency]
            total_amount = multiplier_day1[currency] * multiplier_day2
            if total_amount > max_amount:
                max_amount = total_amount
        
        return max_amount