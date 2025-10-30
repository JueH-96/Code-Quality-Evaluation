class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float],
                  pairs2: List[List[str]], rates2: List[float]) -> float:
        from math import isclose
        
        # STEP 1: Collect all unique currencies (including the initial one).
        currencies = set()
        currencies.add(initialCurrency)
        for (c1, c2) in pairs1:
            currencies.add(c1)
            currencies.add(c2)
        for (c1, c2) in pairs2:
            currencies.add(c1)
            currencies.add(c2)
        currencies = list(currencies)
        
        # Map each currency to an index.
        idx_map = {cur: i for i, cur in enumerate(currencies)}
        n = len(currencies)
        
        # Initialize adjacency matrices for day1 and day2.
        # We'll store the "best" multiplicative rates, so use 0 for "no direct connection"
        # and set diagonal to 1.0 for same-currency rates.
        graph1 = [[0.0]*n for _ in range(n)]
        graph2 = [[0.0]*n for _ in range(n)]
        for i in range(n):
            graph1[i][i] = 1.0
            graph2[i][i] = 1.0
            
        # STEP 2: Fill adjacency for day1.
        for (c1, c2), r in zip(pairs1, rates1):
            i1, i2 = idx_map[c1], idx_map[c2]
            graph1[i1][i2] = r
            graph1[i2][i1] = 1.0 / r
        
        # STEP 3: Fill adjacency for day2.
        for (c1, c2), r in zip(pairs2, rates2):
            i1, i2 = idx_map[c1], idx_map[c2]
            graph2[i1][i2] = r
            graph2[i2][i1] = 1.0 / r
        
        # STEP 4: Floyd-Warshall on both day's graphs to find max conversion rates.
        # dist_day1[i][j] will be the max rate converting from i -> j on day1
        # dist_day2[i][j] will be the max rate converting from i -> j on day2
        dist_day1 = [row[:] for row in graph1]
        dist_day2 = [row[:] for row in graph2]
        
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    # Day1
                    if dist_day1[i][k] * dist_day1[k][j] > dist_day1[i][j]:
                        dist_day1[i][j] = dist_day1[i][k] * dist_day1[k][j]
                    # Day2
                    if dist_day2[i][k] * dist_day2[k][j] > dist_day2[i][j]:
                        dist_day2[i][j] = dist_day2[i][k] * dist_day2[k][j]
        
        # STEP 5: Find the maximum product of:
        #   (initialCurrency -> someCurrency on day1) * (thatCurrency -> initialCurrency on day2).
        init_idx = idx_map[initialCurrency]
        answer = 0.0
        for c in range(n):
            amount = dist_day1[init_idx][c] * dist_day2[c][init_idx]
            if amount > answer:
                answer = amount
        
        return answer