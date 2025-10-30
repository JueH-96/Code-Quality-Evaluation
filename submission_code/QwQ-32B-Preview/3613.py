class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        # Helper function to build graph including inverses
        def build_graph(pairs, rates, include_inverses=True):
            graph = {}
            for i in range(len(pairs)):
                start, target = pairs[i]
                rate = rates[i]
                if start not in graph:
                    graph[start] = {}
                graph[start][target] = rate
                if include_inverses:
                    if target not in graph:
                        graph[target] = {}
                    graph[target][start] = 1 / rate
            # Include identity conversions
            for currency in graph:
                if currency not in graph[currency]:
                    graph[currency][currency] = 1.0
            return graph
        
        # Build day 1 and day 2 graphs
        graph_day1 = build_graph(pairs1, rates1)
        graph_day2 = build_graph(pairs2, rates2)
        
        # Memoization dictionaries
        memo_day1 = {}
        memo_day2 = {}
        
        # DFS for day 1 to compute max amount of C from initialCurrency
        def dfs_day1(currency):
            if currency == initialCurrency:
                return 1.0
            if currency in memo_day1:
                return memo_day1[currency]
            max_amount = 0.0
            for X in graph_day1:
                if currency in graph_day1[X]:
                    amount_from_X = dfs_day1(X)
                    amount_from_X_to_currency = amount_from_X * graph_day1[X][currency]
                    if amount_from_X_to_currency > max_amount:
                        max_amount = amount_from_X_to_currency
            memo_day1[currency] = max_amount
            return max_amount
        
        # DFS for day 2 to compute max amount of initialCurrency from C
        def dfs_day2(currency):
            if currency == initialCurrency:
                return 1.0
            if currency in memo_day2:
                return memo_day2[currency]
            max_amount = 0.0
            for Y in graph_day2[currency]:
                amount_to_Y = graph_day2[currency][Y]
                amount_from_Y = dfs_day2(Y)
                total_amount = amount_to_Y * amount_from_Y
                if total_amount > max_amount:
                    max_amount = total_amount
            memo_day2[currency] = max_amount
            return max_amount
        
        # Collect all unique currencies
        all_currencies = set()
        for pair in pairs1 + pairs2:
            all_currencies.update(pair)
        all_currencies.add(initialCurrency)
        
        # Compute the maximum amount of initialCurrency
        max_initial = 0.0
        for C in all_currencies:
            A_C = dfs_day1(C)
            B_C = dfs_day2(C)
            total = A_C * B_C
            if total > max_initial:
                max_initial = total
        return max_initial