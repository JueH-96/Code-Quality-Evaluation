from typing import List
from collections import defaultdict, deque

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        def compute_max_rates(graph, source):
            nodes = set()
            for u in graph:
                nodes.add(u)
                for v, r in graph[u]:
                    nodes.add(v)
            max_rates = defaultdict(float)
            for node in nodes:
                max_rates[node] = 0.0
            max_rates[source] = 1.0
            queue = deque([source])
            while queue:
                u = queue.popleft()
                for v, r in graph[u]:
                    new_rate = max_rates[u] * r
                    if new_rate > max_rates[v]:
                        max_rates[v] = new_rate
                        queue.append(v)
            return max_rates
        
        # Build day1's graph
        graph1 = defaultdict(list)
        for i in range(len(pairs1)):
            s, t = pairs1[i]
            r = rates1[i]
            graph1[s].append((t, r))
            graph1[t].append((s, 1.0 / r))
        max_day1 = compute_max_rates(graph1, initialCurrency)
        
        # Build day2's reversed graph
        graph2_reversed = defaultdict(list)
        for i in range(len(pairs2)):
            s, t = pairs2[i]
            r = rates2[i]
            graph2_reversed[t].append((s, r))
            graph2_reversed[s].append((t, 1.0 / r))
        max_day2_reversed = compute_max_rates(graph2_reversed, initialCurrency)
        
        # Calculate the maximum total
        max_total = 0.0
        for currency in max_day1:
            current = max_day1[currency]
            rate = max_day2_reversed.get(currency, 0.0)
            total = current * rate
            if total > max_total:
                max_total = total
        return max_total