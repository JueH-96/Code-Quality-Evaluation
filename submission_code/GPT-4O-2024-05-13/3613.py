class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        from collections import defaultdict
        import math

        def build_graph(pairs, rates):
            graph = defaultdict(dict)
            for (src, dst), rate in zip(pairs, rates):
                graph[src][dst] = rate
                graph[dst][src] = 1 / rate
            return graph

        def floyd_warshall(graph):
            currencies = list(graph.keys())
            dist = {cur: {cur: 1.0 for cur in currencies} for cur in currencies}

            for cur in currencies:
                for neighbor in graph[cur]:
                    dist[cur][neighbor] = graph[cur][neighbor]

            for k in currencies:
                for i in currencies:
                    for j in currencies:
                        if dist[i][j] < dist[i][k] * dist[k][j]:
                            dist[i][j] = dist[i][k] * dist[k][j]
            return dist

        graph1 = build_graph(pairs1, rates1)
        graph2 = build_graph(pairs2, rates2)

        dist1 = floyd_warshall(graph1)
        dist2 = floyd_warshall(graph2)

        max_amount = 1.0
        for currency in dist1[initialCurrency]:
            if currency in dist2:
                amount = dist1[initialCurrency][currency] * dist2[currency][initialCurrency]
                max_amount = max(max_amount, amount)

        return max_amount