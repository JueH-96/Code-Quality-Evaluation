from typing import List

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float],
                  pairs2: List[List[str]], rates2: List[float]) -> float:
        # Gather all currencies
        currencies = {initialCurrency}
        for (u, v) in pairs1 + pairs2:
            currencies.add(u)
            currencies.add(v)
        # Map currency to index
        idx = {c: i for i, c in enumerate(currencies)}
        n = len(currencies)
        
        # Initialize adjacency matrices for day1 and day2
        # mat1[i][j] = max rate to convert i->j on day1
        # mat2 similarly for day2
        mat1 = [[0.0]*n for _ in range(n)]
        mat2 = [[0.0]*n for _ in range(n)]
        for i in range(n):
            mat1[i][i] = 1.0
            mat2[i][i] = 1.0
        
        # Fill direct edges for day1
        for (u, v), r in zip(pairs1, rates1):
            a, b = idx[u], idx[v]
            mat1[a][b] = max(mat1[a][b], r)
            mat1[b][a] = max(mat1[b][a], 1.0 / r)
        # Fill direct edges for day2
        for (u, v), r in zip(pairs2, rates2):
            a, b = idx[u], idx[v]
            mat2[a][b] = max(mat2[a][b], r)
            mat2[b][a] = max(mat2[b][a], 1.0 / r)
        
        # Floyd-Warshall (max-product) for day1
        for k in range(n):
            for i in range(n):
                if mat1[i][k] == 0: continue
                for j in range(n):
                    if mat1[k][j] == 0: continue
                    prod = mat1[i][k] * mat1[k][j]
                    if prod > mat1[i][j]:
                        mat1[i][j] = prod
        # Floyd-Warshall for day2
        for k in range(n):
            for i in range(n):
                if mat2[i][k] == 0: continue
                for j in range(n):
                    if mat2[k][j] == 0: continue
                    prod = mat2[i][k] * mat2[k][j]
                    if prod > mat2[i][j]:
                        mat2[i][j] = prod
        
        # Compute the best sequence: day1 initial->c, then day2 c->initial
        start = idx[initialCurrency]
        best = 1.0
        for c in range(n):
            amt_day1 = mat1[start][c]
            if amt_day1 == 0: 
                continue
            amt_day2 = mat2[c][start]
            if amt_day2 == 0:
                continue
            best = max(best, amt_day1 * amt_day2)
        
        return best