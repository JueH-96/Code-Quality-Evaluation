from typing import List

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        currencies = set()
        currencies.add(initialCurrency)
        for u, v in pairs1:
            currencies.add(u)
            currencies.add(v)
        for u, v in pairs2:
            currencies.add(u)
            currencies.add(v)
        currencies = list(currencies)
        n = len(currencies)
        
        edges1 = []
        for i in range(len(pairs1)):
            u, v = pairs1[i]
            r = rates1[i]
            edges1.append((u, v, r))
            edges1.append((v, u, 1.0 / r))
        
        A1 = {c: 0.0 for c in currencies}
        A1[initialCurrency] = 1.0
        
        for _ in range(n - 1):
            updated = False
            for u, v, w in edges1:
                new_val = A1[u] * w
                if new_val > A1[v]:
                    A1[v] = new_val
                    updated = True
            if not updated:
                break
        
        edges2 = []
        for i in range(len(pairs2)):
            u, v = pairs2[i]
            r = rates2[i]
            edges2.append((u, v, r))
            edges2.append((v, u, 1.0 / r))
        
        g = {c: 0.0 for c in currencies}
        g[initialCurrency] = 1.0
        
        for _ in range(n - 1):
            updated = False
            for u, v, w in edges2:
                new_val = w * g[v]
                if new_val > g[u]:
                    g[u] = new_val
                    updated = True
            if not updated:
                break
        
        ans = 0.0
        for c in currencies:
            candidate = A1[c] * g[c]
            if candidate > ans:
                ans = candidate
        
        return ans