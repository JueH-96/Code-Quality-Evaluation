class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        from collections import defaultdict

        # Function to build graph and compute exchange rates via DFS
        def build_amounts(start_currency, pairs, rates):
            graph = defaultdict(list)
            currencies = set()
            for (a, b), rate in zip(pairs, rates):
                graph[a].append((b, rate))
                graph[b].append((a, 1.0 / rate))
                currencies.add(a)
                currencies.add(b)
            amounts = {}
            visited = set()
            def dfs(currency, amount):
                amounts[currency] = amount
                visited.add(currency)
                for neighbor, rate in graph[currency]:
                    if neighbor not in visited:
                        dfs(neighbor, amount * rate)
            dfs(start_currency, 1.0)
            return amounts

        # Compute amount1: amount of each currency reachable on day 1
        amount1 = build_amounts(initialCurrency, pairs1, rates1)
        # Set to keep track of currencies we have at end of day 1
        currencies_day1 = set(amount1.keys())
        max_amount = 0.0

        # For each currency we have at end of day 1, compute amount in initialCurrency after day 2
        for currency in currencies_day1:
            # For each such currency, compute exchange rate to initialCurrency on day2
            # Build amounts from this currency to others on day 2
            amount2 = {}
            visited = set()
            graph2 = defaultdict(list)
            for (a, b), rate in zip(pairs2, rates2):
                graph2[a].append((b, rate))
                graph2[b].append((a, 1.0 / rate))
            found_initial_currency = False
            def dfs(currency2, amount):
                if currency2 == initialCurrency:
                    amount2[initialCurrency] = amount
                    return True
                visited.add(currency2)
                for neighbor, rate in graph2[currency2]:
                    if neighbor not in visited:
                        if dfs(neighbor, amount * rate):
                            return True
                return False
            if dfs(currency, 1.0):
                total_amount = amount1[currency] * amount2[initialCurrency]
                max_amount = max(max_amount, total_amount)
            else:
                # Cannot convert this currency back to initialCurrency
                pass

        # Also consider the case where we do nothing on day2
        if initialCurrency in amount1:
            max_amount = max(max_amount, amount1[initialCurrency])

        return max_amount