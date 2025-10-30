class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], 
                  pairs2: List[List[str]], rates2: List[float]) -> float:
        
        from collections import deque
        
        # Build an adjacency dictionary for a given list of pairs and rates
        # graph[currency1][currency2] = conversion rate (currency1 -> currency2)
        def build_graph(pairs, rates):
            graph = {}
            for (start, end), r in zip(pairs, rates):
                graph.setdefault(start, {})[end] = r
                graph.setdefault(end, {})[start] = 1.0 / r
            return graph
        
        # Returns a dictionary of conversion rates from start_currency to every reachable currency
        # amounts[c] = (start_currency -> c)
        def compute_amounts(graph, start_currency):
            amounts = {start_currency: 1.0}
            queue = deque([start_currency])
            while queue:
                current = queue.popleft()
                for neighbor, rate in graph.get(current, {}).items():
                    if neighbor not in amounts:
                        amounts[neighbor] = amounts[current] * rate
                        queue.append(neighbor)
            return amounts
        
        # Build graphs for day 1 and day 2
        day1_graph = build_graph(pairs1, rates1)
        day2_graph = build_graph(pairs2, rates2)

        # Compute all possible amounts from initialCurrency after day 1
        day1_amounts = compute_amounts(day1_graph, initialCurrency)
        
        # For each currency c that we can obtain on day 1, compute how much of initialCurrency
        # we could get after day 2 if we start day 2 holding currency c.
        # We'll do a BFS from c in the day2 graph to find ratio c->initialCurrency
        max_amount = 1.0  # At least we can keep the 1.0 of initialCurrency (zero conversions)
        for c, amount_after_day1 in day1_amounts.items():
            day2_amounts = compute_amounts(day2_graph, c)
            if initialCurrency in day2_amounts:
                # day2_amounts[initialCurrency] is the ratio (c -> initialCurrency)
                candidate = amount_after_day1 * day2_amounts[initialCurrency]
                max_amount = max(max_amount, candidate)
        
        return max_amount