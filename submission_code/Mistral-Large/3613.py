from typing import List, Dict
from collections import defaultdict

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        def build_graph(pairs, rates):
            graph = defaultdict(dict)
            for (start, target), rate in zip(pairs, rates):
                graph[start][target] = rate
                graph[target][start] = 1 / rate
            return graph

        def dfs(currency, graph, visited, amount):
            if currency in visited:
                return visited[currency]
            max_amount = amount
            for next_currency, rate in graph[currency].items():
                max_amount = max(max_amount, dfs(next_currency, graph, visited, amount * rate))
            visited[currency] = max_amount
            return max_amount

        graph1 = build_graph(pairs1, rates1)
        graph2 = build_graph(pairs2, rates2)

        max_amount_day1 = dfs(initialCurrency, graph1, {}, 1.0)
        max_amount_day2 = 0.0

        for currency in graph1:
            for next_currency, rate in graph1[currency].items():
                max_amount_day2 = max(max_amount_day2, dfs(next_currency, graph2, {}, max_amount_day1 * rate))

        return max_amount_day2