class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        def solve(initial_amount, initial_currency, pairs, rates):
            graph = {}
            for i in range(len(pairs)):
                start, end = pairs[i]
                rate = rates[i]
                if start not in graph:
                    graph[start] = []
                if end not in graph:
                    graph[end] = []
                graph[start].append((end, rate))
                graph[end].append((start, 1 / rate))

            max_amount = 0.0

            def dfs(currency, amount, visited):
                nonlocal max_amount
                if currency == initialCurrency:
                    max_amount = max(max_amount, amount)

                for neighbor, rate in graph.get(currency, []):
                    if neighbor not in visited:
                        new_visited = visited.copy()
                        new_visited.add(neighbor)
                        dfs(neighbor, amount * rate, new_visited)

            dfs(initial_currency, initial_amount, {initial_currency})
            return max_amount

        amount1 = solve(1.0, initialCurrency, pairs1, rates1)
        
        graph1 = {}
        for i in range(len(pairs1)):
            start, end = pairs1[i]
            rate = rates1[i]
            if start not in graph1:
                graph1[start] = []
            if end not in graph1:
                graph1[end] = []
            graph1[start].append((end, rate))
            graph1[end].append((start, 1 / rate))
        
        currencies = set([initialCurrency])
        for p in pairs1:
            currencies.add(p[0])
            currencies.add(p[1])
        
        amounts = {initialCurrency: 1.0}
        for c in currencies:
            if c != initialCurrency:
                amounts[c] = 0.0
        
        
        def dfs1(currency, amount, visited):
            amounts[currency] = max(amounts[currency], amount)
            for neighbor, rate in graph1.get(currency, []):
                if neighbor not in visited:
                    new_visited = visited.copy()
                    new_visited.add(neighbor)
                    dfs1(neighbor, amount * rate, new_visited)
        
        dfs1(initialCurrency, 1.0, {initialCurrency})
        
        max_final_amount = 0.0
        for currency, amount in amounts.items():
            max_final_amount = max(max_final_amount, solve(amount, currency, pairs2, rates2))
        
        return max_final_amount