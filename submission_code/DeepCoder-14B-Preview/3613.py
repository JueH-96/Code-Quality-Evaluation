from typing import List

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        # Collect all unique currencies
        currencies = set()
        currencies.add(initialCurrency)
        for pair in pairs1:
            currencies.add(pair[0])
            currencies.add(pair[1])
        for pair in pairs2:
            currencies.add(pair[0])
            currencies.add(pair[1])
        currencies = list(currencies)
        
        # Build edges for day 1 and day 2 including reverse conversions
        edges_day1 = []
        for i in range(len(pairs1)):
            a = pairs1[i][0]
            b = pairs1[i][1]
            r = rates1[i]
            edges_day1.append((a, b, r))
            edges_day1.append((b, a, 1.0 / r))
        
        edges_day2 = []
        for i in range(len(pairs2)):
            a = pairs2[i][0]
            b = pairs2[i][1]
            r = rates2[i]
            edges_day2.append((a, b, r))
            edges_day2.append((b, a, 1.0 / r))
        
        # Helper function to compute maximum exchange rates using Floyd-Warshall algorithm
        def compute_max_rates(currencies, edges):
            n = len(currencies)
            index = {c: i for i, c in enumerate(currencies)}
            # Initialize the matrix with 1.0 for same currencies and 0.0 otherwise
            matrix = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
            # Add all edges
            for a, b, r in edges:
                i = index[a]
                j = index[b]
                if matrix[i][j] < r:
                    matrix[i][j] = r
            # Floyd-Warshall algorithm to find maximum path products
            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        if matrix[i][k] > 0 and matrix[k][j] > 0:
                            if matrix[i][j] < matrix[i][k] * matrix[k][j]:
                                matrix[i][j] = matrix[i][k] * matrix[k][j]
            return matrix
        
        # Compute maximum rates for both days
        day1_matrix = compute_max_rates(currencies, edges_day1)
        day2_matrix = compute_max_rates(currencies, edges_day2)
        
        # Find the maximum product of day1 and day2 rates to get the initial currency back
        max_total = 0.0
        initial_index = currencies.index(initialCurrency)
        for x in currencies:
            x_index = currencies.index(x)
            day1_rate = day1_matrix[initial_index][x_index]
            day2_rate = day2_matrix[x_index][initial_index]
            total = day1_rate * day2_rate
            if total > max_total:
                max_total = total
        
        # The maximum possible amount is at least the initial amount (1.0)
        # Ensure that we return the value rounded to 5 decimal places
        return round(max_total, 5)