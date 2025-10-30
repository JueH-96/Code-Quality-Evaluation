from typing import List

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        # Build the graphs for both days
        dist1, curr_list1, index_map1 = self.build_graph(pairs1, rates1, initialCurrency)
        dist2, curr_list2, index_map2 = self.build_graph(pairs2, rates2, initialCurrency)
        
        # Get the index of the initial currency in both days' graphs
        initial_idx1 = index_map1[initialCurrency]
        initial_idx2 = index_map2[initialCurrency]
        
        # Calculate best rates from initial to all currencies in day 1
        max_day1 = {}
        for i, curr in enumerate(curr_list1):
            max_day1[curr] = dist1[initial_idx1][i]
        
        # Calculate best rates from all currencies to initial in day 2
        max_day2 = {}
        for i, curr in enumerate(curr_list2):
            max_day2[curr] = dist2[i][initial_idx2]
        
        # Compute the maximum product
        result = 1.0  # At least the initial amount
        for curr in max_day1:
            if curr in max_day2:
                product = max_day1[curr] * max_day2[curr]
                if product > result:
                    result = product
        return result
    
    def build_graph(self, pairs: List[List[str]], rates: List[float], initialCurrency: str):
        # Collect all currencies involved in the pairs and the initialCurrency
        currencies = set()
        for a, b in pairs:
            currencies.add(a)
            currencies.add(b)
        currencies.add(initialCurrency)
        currencies = list(currencies)
        n = len(currencies)
        index_map = {curr: idx for idx, curr in enumerate(currencies)}
        
        # Initialize distance matrix
        dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 1.0
        
        # Populate the graph with direct and reverse edges
        for (a, b), rate in zip(pairs, rates):
            i = index_map[a]
            j = index_map[b]
            if dist[i][j] < rate:
                dist[i][j] = rate
            rev_rate = 1.0 / rate
            if dist[j][i] < rev_rate:
                dist[j][i] = rev_rate
        
        # Floyd-Warshall algorithm to find the best rates
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] > 0 and dist[k][j] > 0:
                        product = dist[i][k] * dist[k][j]
                        if product > dist[i][j]:
                            dist[i][j] = product
        
        return dist, currencies, index_map