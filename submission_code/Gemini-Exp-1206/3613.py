from collections import defaultdict
from typing import List

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        graph1 = defaultdict(dict)
        graph2 = defaultdict(dict)

        for (start, end), rate in zip(pairs1, rates1):
            graph1[start][end] = rate
            graph1[end][start] = 1 / rate

        for (start, end), rate in zip(pairs2, rates2):
            graph2[start][end] = rate
            graph2[end][start] = 1 / rate

        def bfs(graph, start_currency):
            q = [(start_currency, 1.0)]
            max_rates = defaultdict(lambda: 0.0)
            max_rates[start_currency] = 1.0

            while q:
                curr, rate = q.pop(0)
                for neighbor, edge_rate in graph[curr].items():
                    new_rate = rate * edge_rate
                    if new_rate > max_rates[neighbor]:
                        max_rates[neighbor] = new_rate
                        q.append((neighbor, new_rate))
            return max_rates

        max_rates_day1 = bfs(graph1, initialCurrency)
        
        max_final_rate = 0.0
        for currency, rate_day1 in max_rates_day1.items():
            max_rates_day2 = bfs(graph2, currency)
            if initialCurrency in max_rates_day2:
                max_final_rate = max(max_final_rate, rate_day1 * max_rates_day2[initialCurrency])

        return max_final_rate