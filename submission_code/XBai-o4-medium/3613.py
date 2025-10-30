from typing import List

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        # Process day 1
        edges1 = []
        currencies1 = set()
        currencies1.add(initialCurrency)
        for i in range(len(pairs1)):
            a, b = pairs1[i][0], pairs1[i][1]
            currencies1.add(a)
            currencies1.add(b)
            rate = rates1[i]
            edges1.append((a, b, rate))
            edges1.append((b, a, 1.0 / rate))
        
        # Bellman-Ford for day 1
        dist_day1 = {currency: 0.0 for currency in currencies1}
        dist_day1[initialCurrency] = 1.0
        for _ in range(len(currencies1) - 1):
            updated = False
            for (u, v, rate) in edges1:
                if dist_day1[u] * rate > dist_day1[v]:
                    dist_day1[v] = dist_day1[u] * rate
                    updated = True
            if not updated:
                break
        
        # Process day 2
        edges2 = []
        for i in range(len(pairs2)):
            a, b = pairs2[i][0], pairs2[i][1]
            rate = rates2[i]
            edges2.append((a, b, rate))
            edges2.append((b, a, 1.0 / rate))
        
        # Collect currencies for day 2
        currencies2 = set()
        for pair in pairs2:
            currencies2.add(pair[0])
            currencies2.add(pair[1])
        currencies2.update(dist_day1.keys())
        
        # Bellman-Ford for day 2
        dist_day2 = dist_day1.copy()
        for _ in range(len(currencies2) - 1):
            updated = False
            for (u, v, rate) in edges2:
                if dist_day2[u] * rate > dist_day2[v]:
                    dist_day2[v] = dist_day2[u] * rate
                    updated = True
            if not updated:
                break
        
        return dist_day2.get(initialCurrency, 1.0)