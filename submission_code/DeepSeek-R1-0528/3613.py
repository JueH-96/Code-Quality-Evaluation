from collections import defaultdict

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        V1 = set()
        V1.add(initialCurrency)
        graph1 = defaultdict(list)
        for (u, v), r in zip(pairs1, rates1):
            V1.add(u)
            V1.add(v)
            graph1[u].append((v, r))
            graph1[v].append((u, 1.0 / r))
        
        max_amount1 = {c: 0.0 for c in V1}
        max_amount1[initialCurrency] = 1.0
        n1 = len(V1)
        for _ in range(n1 - 1):
            updated = False
            for u in V1:
                current = max_amount1[u]
                if current == 0.0:
                    continue
                for (v, rate) in graph1[u]:
                    new_amt = current * rate
                    if new_amt > max_amount1[v]:
                        max_amount1[v] = new_amt
                        updated = True
            if not updated:
                break
        
        V2 = set()
        V2.add(initialCurrency)
        graph2 = defaultdict(list)
        for (u, v), r in zip(pairs2, rates2):
            V2.add(u)
            V2.add(v)
            graph2[u].append((v, r))
            graph2[v].append((u, 1.0 / r))
        
        h = {c: 0.0 for c in V2}
        h[initialCurrency] = 1.0
        n2 = len(V2)
        for _ in range(n2 - 1):
            updated = False
            for u in V2:
                for (v, rate) in graph2[u]:
                    candidate = rate * h[v]
                    if candidate > h[u]:
                        h[u] = candidate
                        updated = True
            if not updated:
                break
        
        ans = 0.0
        for currency in V1:
            if currency in V2:
                candidate = max_amount1[currency] * h[currency]
                if candidate > ans:
                    ans = candidate
        return ans