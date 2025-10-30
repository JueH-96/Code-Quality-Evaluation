from typing import List

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        # Compute edges for day1
        edges1 = []
        for i in range(len(pairs1)):
            a, b = pairs1[i]
            rate = rates1[i]
            edges1.append((a, b, rate))
            edges1.append((b, a, 1.0 / rate))
        
        # Compute day1_max
        day1_max = self.compute_max(edges1, initialCurrency, 1.0)
        
        # Compute edges for day2
        edges2 = []
        for i in range(len(pairs2)):
            a, b = pairs2[i]
            rate = rates2[i]
            edges2.append((a, b, rate))
            edges2.append((b, a, 1.0 / rate))
        
        # Compute max_day2_to_initial
        max_day2_to_initial = {}
        currencies_day2 = set()
        for a, b, r in edges2:
            currencies_day2.add(a)
            currencies_day2.add(b)
        
        # First compute for all currencies in edges2
        for X in currencies_day2:
            max_values_X = self.compute_max(edges2, X, 1.0)
            max_day2_to_initial[X] = max_values_X.get(initialCurrency, 0.0)
        
        # Handle currencies not in edges2's currencies but present in day1_max
        currencies_day2_set = currencies_day2
        for C in day1_max:
            if C not in currencies_day2_set:
                if C == initialCurrency:
                    max_day2_to_initial[C] = 1.0
                else:
                    max_day2_to_initial[C] = 0.0
        
        # Compute the maximum over all currencies in day1_max
        max_total = 0.0
        for C in day1_max:
            current_amount = day1_max[C]
            conversion_rate = max_day2_to_initial.get(C, 0.0)
            candidate = current_amount * conversion_rate
            if candidate > max_total:
                max_total = candidate
        
        return max_total
    
    def compute_max(self, edges, start_currency, start_amount):
        currencies = set()
        for u, v, r in edges:
            currencies.add(u)
            currencies.add(v)
        currencies = list(currencies)
        max_values = {c: 0.0 for c in currencies}
        max_values[start_currency] = start_amount
        changed = True
        while changed:
            changed = False
            for u, v, r in edges:
                if max_values[u] * r > max_values[v]:
                    new_val = max_values[u] * r
                    if new_val > max_values[v]:
                        max_values[v] = new_val
                        changed = True
        return max_values