from typing import List
from collections import defaultdict

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        def build_graph(pairs, rates):
            graph = defaultdict(dict)
            for (start, end), rate in zip(pairs, rates):
                graph[start][end] = rate
                graph[end][start] = 1 / rate
            return graph

        def dfs(graph, start, end, visited):
            if start == end:
                return 1.0
            visited.add(start)
            max_rate = 0.0
            for neighbor in graph[start]:
                if neighbor not in visited:
                    max_rate = max(max_rate, graph[start][neighbor] * dfs(graph, neighbor, end, visited))
            visited.remove(start)
            return max_rate

        graph1 = build_graph(pairs1, rates1)
        graph2 = build_graph(pairs2, rates2)

        max_amount = 1.0
        for currency in graph1:
            rate1 = dfs(graph1, initialCurrency, currency, set())
            for currency2 in graph2:
                rate2 = dfs(graph2, currency, initialCurrency, set())
                max_amount = max(max_amount, rate1 * rate2)

        return max_amount

# Example usage:
# solution = Solution()
# print(solution.maxAmount("EUR", [["EUR","USD"],["USD","JPY"]], [2.0,3.0], [["JPY","USD"],["USD","CHF"],["CHF","EUR"]], [4.0,5.0,6.0]))  # Output: 720.00000