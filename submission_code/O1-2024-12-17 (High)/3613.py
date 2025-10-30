class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], 
                  pairs2: List[List[str]], rates2: List[float]) -> float:
        
        # Gathers all currencies mentioned plus the initialCurrency.
        currencies = set()
        currencies.add(initialCurrency)
        for (a, b) in pairs1:
            currencies.add(a)
            currencies.add(b)
        for (a, b) in pairs2:
            currencies.add(a)
            currencies.add(b)
        currencies = list(currencies)  # We'll just keep them in a list for iteration.
        
        # Build edges for day1 and day2 as (u, v, rate) meaning 1 unit of u -> rate units of v.
        def buildEdges(pairs, rates):
            edges = []
            for (u, v), r in zip(pairs, rates):
                edges.append((u, v, r))
                edges.append((v, u, 1.0 / r))
            return edges
        
        day1Edges = buildEdges(pairs1, rates1)
        day2Edges = buildEdges(pairs2, rates2)
        
        # Bellman-Ford style routine to get maximum amounts from a single source to all.
        # Here, dist[c] represents the best achievable amount of currency c starting from "start" with 1.0 units.
        def bellmanFord(edges, start, all_currencies):
            dist = {c: 0.0 for c in all_currencies}
            dist[start] = 1.0
            n = len(all_currencies)
            
            # Relax up to n-1 times
            for _ in range(n - 1):
                updated = False
                for (u, v, r) in edges:
                    if dist[u] * r > dist[v]:
                        dist[v] = dist[u] * r
                        updated = True
                if not updated:
                    break
            return dist
        
        # For day 1, find the best we can do for each currency c, starting with 1.0 of initialCurrency.
        day1_amount = bellmanFord(day1Edges, initialCurrency, currencies)
        
        # For day 2, we need a function that given a start c, returns the maximum amount of initialCurrency we can get.
        # We'll again use Bellman-Ford but now we interpret "start" as having 1.0 of that currency,
        # and want dist[initialCurrency].
        def day2_max_from(c):
            dist = {x: 0.0 for x in currencies}
            dist[c] = 1.0
            n = len(currencies)
            for _ in range(n - 1):
                updated = False
                for (u, v, r) in day2Edges:
                    if dist[u] * r > dist[v]:
                        dist[v] = dist[u] * r
                        updated = True
                if not updated:
                    break
            return dist[initialCurrency]
        
        # Compute, for each currency c, how many units of initialCurrency we can get if we start day 2 with 1.0 of c.
        day2_amount = {}
        for c in currencies:
            day2_amount[c] = day2_max_from(c)
        
        # The answer is the maximum over all currencies c of (amount of c on day1) * (amount of initialCurrency on day2 from c).
        answer = 0.0
        for c in currencies:
            possible_final = day1_amount[c] * day2_amount[c]
            if possible_final > answer:
                answer = possible_final
        
        return answer