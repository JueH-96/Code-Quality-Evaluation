class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        from collections import defaultdict
        import heapq

        def build_graph(pairs, rates):
            graph = defaultdict(dict)
            for (src, dst), rate in zip(pairs, rates):
                graph[src][dst] = rate
                graph[dst][src] = 1 / rate
            return graph

        def max_currency_from(graph, start_currency):
            # Use Bellman-Ford algorithm to find the maximum conversion rate to every other currency from start_currency
            max_currency = defaultdict(float)
            max_currency[start_currency] = 1.0
            currencies = list(graph.keys())
            
            # Relax edges up to len(currencies) - 1 times
            for _ in range(len(currencies) - 1):
                for u in currencies:
                    for v in graph[u]:
                        if max_currency[v] < max_currency[u] * graph[u][v]:
                            max_currency[v] = max_currency[u] * graph[u][v]
            return max_currency

        # Build graphs for day 1 and day 2
        graph1 = build_graph(pairs1, rates1)
        graph2 = build_graph(pairs2, rates2)

        # Calculate maximum currency values from initialCurrency after day 1
        max_after_day1 = max_currency_from(graph1, initialCurrency)

        # Calculate maximum currency values from each currency after day 2
        max_final = defaultdict(float)
        for currency in max_after_day1:
            max_from_currency_day2 = max_currency_from(graph2, currency)
            for target_currency in max_from_currency_day2:
                max_final[target_currency] = max(max_final[target_currency], max_after_day1[currency] * max_from_currency_day2[target_currency])

        # The answer is the maximum amount of initialCurrency we can get after both days
        return max_final[initialCurrency]