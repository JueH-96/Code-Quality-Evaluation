from typing import List

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        def build_max_rates(pairs, rates, initial):
            currencies = set()
            currencies.add(initial)
            for A, B in pairs:
                currencies.add(A)
                currencies.add(B)
            currencies = sorted(currencies)
            n = len(currencies)
            c_to_idx = {c: i for i, c in enumerate(currencies)}
            max_rate = [[0.0 for _ in range(n)] for _ in range(n)]
            for i in range(n):
                max_rate[i][i] = 1.0
            for (A, B), r in zip(pairs, rates):
                i = c_to_idx[A]
                j = c_to_idx[B]
                if r > max_rate[i][j]:
                    max_rate[i][j] = r
                reverse_r = 1.0 / r
                if reverse_r > max_rate[j][i]:
                    max_rate[j][i] = reverse_r
            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        if max_rate[i][k] * max_rate[k][j] > max_rate[i][j]:
                            max_rate[i][j] = max_rate[i][k] * max_rate[k][j]
            return currencies, c_to_idx, max_rate
        
        currencies1, c_to_idx1, max_rate1 = build_max_rates(pairs1, rates1, initialCurrency)
        max_day1 = {}
        initial_idx1 = c_to_idx1[initialCurrency]
        for i, c in enumerate(currencies1):
            max_day1[c] = max_rate1[initial_idx1][i]
        
        currencies2, c_to_idx2, max_rate2 = build_max_rates(pairs2, rates2, initialCurrency)
        max_day2 = {}
        initial_idx2 = c_to_idx2[initialCurrency]
        for i, c in enumerate(currencies2):
            max_day2[c] = max_rate2[i][initial_idx2]
        
        max_total = 0.0
        for c in max_day1:
            if c in max_day2:
                product = max_day1[c] * max_day2[c]
                if product > max_total:
                    max_total = product
        
        return round(max_total, 5)