class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        def build_graph(pairs, rates):
            graph = {}
            for (start, target), rate in zip(pairs, rates):
                if start not in graph:
                    graph[start] = {}
                if target not in graph:
                    graph[target] = {}
                graph[start][target] = rate
                graph[target][start] = 1 / rate
            return graph

        def dfs(currency, amount, graph, visited):
            visited.add(currency)
            max_amount = amount
            for next_currency, rate in graph[currency].items():
                if next_currency not in visited:
                    max_amount = max(max_amount, dfs(next_currency, amount * rate, graph, visited))
            visited.remove(currency)
            return max_amount

        graph1 = build_graph(pairs1, rates1)
        graph2 = build_graph(pairs2, rates2)

        # Day 1
        max_day1 = dfs(initialCurrency, 1.0, graph1, set())

        # Day 2
        all_currencies = set(graph1.keys()) | set(graph2.keys())
        max_final = max_day1

        for currency in all_currencies:
            amount_day2 = dfs(currency, max_day1, graph2, set())
            max_final = max(max_final, amount_day2)

        return max_final