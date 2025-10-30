from typing import List

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        # Step 1: Collect all unique currencies
        currencies = set([initialCurrency])
        for pair in pairs1 + pairs2:
            currencies.update(pair)
        currencies = list(currencies)
        currency_to_idx = {currency: idx for idx, currency in enumerate(currencies)}
        n = len(currencies)
        
        # Step 2: Initialize rate matrices for day1 and day2
        rate1 = [[0.0 for _ in range(n)] for _ in range(n)]
        rate2 = [[0.0 for _ in range(n)] for _ in range(n)]
        
        # Set self rates to 1
        src_idx = currency_to_idx[initialCurrency]
        rate1[src_idx][src_idx] = 1.0
        rate2[src_idx][src_idx] = 1.0
        
        # Step 3: Populate rate1 with day1 conversion rates
        for (s, t), r in zip(pairs1, rates1):
            s_idx = currency_to_idx[s]
            t_idx = currency_to_idx[t]
            if r > rate1[s_idx][t_idx]:
                rate1[s_idx][t_idx] = r
            inverse_r = 1.0 / r
            if inverse_r > rate1[t_idx][s_idx]:
                rate1[t_idx][s_idx] = inverse_r
        
        # Step 4: Perform Floyd-Warshall for day1 to find max rates from initialCurrency
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if rate1[i][k] * rate1[k][j] > rate1[i][j]:
                        rate1[i][j] = rate1[i][k] * rate1[k][j]
        
        # Step 5: Populate rate2 with day2 conversion rates
        for (s, t), r in zip(pairs2, rates2):
            s_idx = currency_to_idx[s]
            t_idx = currency_to_idx[t]
            if r > rate2[s_idx][t_idx]:
                rate2[s_idx][t_idx] = r
            inverse_r = 1.0 / r
            if inverse_r > rate2[t_idx][s_idx]:
                rate2[t_idx][s_idx] = inverse_r
        
        # Step 6: Perform Floyd-Warshall for day2 to find max rates to initialCurrency
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if rate2[i][k] * rate2[k][j] > rate2[i][j]:
                        rate2[i][j] = rate2[i][k] * rate2[k][j]
        
        # Step 7: Calculate the maximum amount by considering all possible intermediate currencies
        max_amount = 1.0  # At least the initial amount without any conversions
        for c in range(n):
            amount_day1 = rate1[src_idx][c]
            amount_day2 = rate2[c][src_idx]
            if amount_day1 > 0 and amount_day2 > 0:
                total = amount_day1 * amount_day2
                if total > max_amount:
                    max_amount = total
        
        return max_amount