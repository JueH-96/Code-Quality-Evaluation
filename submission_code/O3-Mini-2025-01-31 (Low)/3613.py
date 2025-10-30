from typing import List

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        # Helper to build conversion graph and run Floyd Warshall to compute maximum conversion factor.
        def build_max_rate(pairs: List[List[str]], rates: List[float]):
            # Collect all currencies available in these pairs.
            currencies = set()
            for a, b in pairs:
                currencies.add(a)
                currencies.add(b)
            # Also, include initialCurrency in case it is isolated, though in that day no conversion is done.
            currencies.add(initialCurrency)
            currencies = list(currencies)
            n = len(currencies)
            # Create a mapping from currency to index.
            idx = {curr: i for i, curr in enumerate(currencies)}
            # Initialize conversion matrix with 0's; diagonal = 1.0
            conv = [[0.0 for _ in range(n)] for _ in range(n)]
            for i in range(n):
                conv[i][i] = 1.0
            # Add edges for given conversions and their inverses.
            for (a, b), rate in zip(pairs, rates):
                i = idx[a]
                j = idx[b]
                # if multiple conversion paths exist, take the max rate
                conv[i][j] = max(conv[i][j], rate)
                conv[j][i] = max(conv[j][i], 1.0 / rate)
            
            # Floyd Warshall style update: 
            # For every intermediate currency k, update 
            # conv[i][j] = max(conv[i][j], conv[i][k] * conv[k][j])
            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        if conv[i][k] * conv[k][j] > conv[i][j]:
                            conv[i][j] = conv[i][k] * conv[k][j]
            return conv, idx

        # Build conversion graphs for day 1 and day 2
        conv1, idx1 = build_max_rate(pairs1, rates1)
        conv2, idx2 = build_max_rate(pairs2, rates2)
        
        # The idea is:
        # On Day 1, convert from initialCurrency to some intermediate currency X.
        # On Day 2, convert from that same currency X back to initialCurrency.
        # Overall factor = conv1[initialCurrency->X] * conv2[X->initialCurrency].
        # We maximize over all possible currencies X.
        max_val = 1.0  # At least doing no conversion yields 1.0 initialCurrency.
        
        # We iterate over currencies that appear in both graphs.
        for currency in idx1:
            if currency in idx2:
                i1 = idx1[initialCurrency]
                j1 = idx1[currency]
                i2 = idx2[currency]
                j2 = idx2[initialCurrency]
                # Multiply conversion factors from day 1 and day 2.
                candidate = conv1[i1][j1] * conv2[i2][j2]
                if candidate > max_val:
                    max_val = candidate
                    
        return max_val
                        
# Example usage:
# sol = Solution()
# print(sol.maxAmount("EUR", [["EUR","USD"],["USD","JPY"]], [2.0,3.0], [["JPY","USD"],["USD","CHF"],["CHF","EUR"]], [4.0,5.0,6.0]))  # Expected 720.0