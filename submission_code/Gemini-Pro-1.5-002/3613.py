class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        graph1 = {}
        for i in range(len(pairs1)):
            u, v = pairs1[i]
            rate = rates1[i]
            if u not in graph1:
                graph1[u] = []
            if v not in graph1:
                graph1[v] = []
            graph1[u].append((v, rate))
            graph1[v].append((u, 1 / rate))

        graph2 = {}
        for i in range(len(pairs2)):
            u, v = pairs2[i]
            rate = rates2[i]
            if u not in graph2:
                graph2[u] = []
            if v not in graph2:
                graph2[v] = []
            graph2[u].append((v, rate))
            graph2[v].append((u, 1 / rate))

        def dfs(graph, start_node, amount):
            max_amount = amount
            visited = set()
            stack = [(start_node, amount)]

            while stack:
                node, current_amount = stack.pop()
                max_amount = max(max_amount, current_amount)

                if node in graph:
                    for neighbor, rate in graph[node]:
                        if (node, neighbor) not in visited:
                            visited.add((node, neighbor))
                            stack.append((neighbor, current_amount * rate))
            return max_amount
        
        day1_max = dfs(graph1, initialCurrency, 1.0)
        day2_max = dfs(graph2, initialCurrency, 1.0)

        
        if graph1 and graph2:
            max_val = 0
            for currency in graph1:
                day1_amount = dfs(graph1, initialCurrency, 1.0)
                if currency in graph1:
                    day1_currency_amount = dfs(graph1, initialCurrency, 1.0) / dfs(graph1, currency, 1.0)
                    day1_to_currency = dfs(graph1, initialCurrency, 1.0)
                    day2_from_currency = dfs(graph2, currency, 1.0)
                    max_val = max(max_val, day1_to_currency * day2_from_currency)
            return max_val
        elif graph1:
            return day1_max
        elif graph2:
            return day2_max
        else:
            return 1.0