from typing import List

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        # Compute day1_max
        currencies_day1 = set()
        for a, b in pairs1:
            currencies_day1.add(a)
            currencies_day1.add(b)
        currencies_day1.add(initialCurrency)
        currencies_day1 = list(currencies_day1)
        
        edges_day1 = []
        for i in range(len(pairs1)):
            a, b = pairs1[i]
            r = rates1[i]
            edges_day1.append((a, b, r))
            edges_day1.append((b, a, 1.0 / r))
        
        day1_max = {currency: 0.0 for currency in currencies_day1}
        day1_max[initialCurrency] = 1.0
        
        for _ in range(len(currencies_day1) - 1):
            for a, b, r in edges_day1:
                if day1_max[a] * r > day1_max[b]:
                    day1_max[b] = day1_max[a] * r
        
        # Compute day2_conversion
        currencies_day2 = set()
        for a, b in pairs2:
            currencies_day2.add(a)
            currencies_day2.add(b)
        currencies_day2.add(initialCurrency)
        currencies_day2 = list(currencies_day2)
        
        edges_day2 = []
        for i in range(len(pairs2)):
            a, b = pairs2[i]
            r = rates2[i]
            edges_day2.append((a, b, r))
            edges_day2.append((b, a, 1.0 / r))
        
        day2_conversion = {currency: 0.0 for currency in currencies_day2}
        day2_conversion[initialCurrency] = 1.0
        
        for _ in range(len(currencies_day2) - 1):
            for a, b, r in edges_day2:
                if day2_conversion[a] < r * day2_conversion[b]:
                    day2_conversion[a] = r * day2_conversion[b]
        
        max_result = 0.0
        for x in day1_max:
            if x in day2_conversion:
                current = day1_max[x] * day2_conversion[x]
            else:
                current = day1_max[x] * 1.0 if x == initialCurrency else 0.0
            if current > max_result:
                max_result = current
        
        return max_result