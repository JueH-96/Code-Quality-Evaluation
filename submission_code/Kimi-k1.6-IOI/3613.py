from typing import List
from collections import deque

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        # Collect all unique currencies
        all_currencies = {initialCurrency}
        for pair in pairs1:
            all_currencies.add(pair[0])
            all_currencies.add(pair[1])
        for pair in pairs2:
            all_currencies.add(pair[0])
            all_currencies.add(pair[1])
        
        # Build adjacency list for day 1
        adj1 = {curr: [] for curr in all_currencies}
        for i in range(len(pairs1)):
            start, end = pairs1[i]
            rate = rates1[i]
            adj1[start].append((end, rate))
            adj1[end].append((start, 1.0 / rate))
        
        # Build adjacency list for day 2
        adj2 = {curr: [] for curr in all_currencies}
        for i in range(len(pairs2)):
            start, end = pairs2[i]
            rate = rates2[i]
            adj2[start].append((end, rate))
            adj2[end].append((start, 1.0 / rate))
        
        # Function to compute maximum values starting from a given currency using an adjacency list
        def compute_max(start, adj):
            max_vals = {curr: 0.0 for curr in adj}
            max_vals[start] = 1.0
            queue = deque([start])
            while queue:
                curr = queue.popleft()
                for neighbor, rate in adj[curr]:
                    new_value = max_vals[curr] * rate
                    if new_value > max_vals[neighbor]:
                        max_vals[neighbor] = new_value
                        queue.append(neighbor)
            return max_vals
        
        # Compute maximum values for day 1 starting from initialCurrency
        value1 = compute_max(initialCurrency, adj1)
        
        # Compute maximum values for day 2 for each currency, targeting initialCurrency
        initial = initialCurrency
        value2 = {}
        for curr in all_currencies:
            temp = compute_max(curr, adj2)
            value2[curr] = temp.get(initial, 0.0)
        
        # Calculate the maximum product of value1 and value2 for all currencies
        max_amount = 0.0
        for curr in all_currencies:
            product = value1[curr] * value2[curr]
            if product > max_amount:
                max_amount = product
        
        return max_amount