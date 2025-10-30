from typing import List

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        currencies = set([initialCurrency])
        for p in pairs1:
            currencies.add(p[0])
            currencies.add(p[1])
        for p in pairs2:
            currencies.add(p[0])
            currencies.add(p[1])
        currencies = list(currencies)
        currency_to_index = {currency: i for i, currency in enumerate(currencies)}
        num_currencies = len(currencies)

        day1_rates_map = {}
        for i in range(len(pairs1)):
            start, target = pairs1[i]
            rate = rates1[i]
            if start not in day1_rates_map:
                day1_rates_map[start] = {}
            day1_rates_map[start][target] = rate
            if target not in day1_rates_map:
                day1_rates_map[target] = {}
            day1_rates_map[target][start] = 1.0 / rate

        day2_rates_map = {}
        for i in range(len(pairs2)):
            start, target = pairs2[i]
            rate = rates2[i]
            if start not in day2_rates_map:
                day2_rates_map[start] = {}
            day2_rates_map[start][target] = rate
            if target not in day2_rates_map:
                day2_rates_map[target] = {}
            day2_rates_map[target][start] = 1.0 / rate

        dp1 = {}
        for currency in currencies:
            dp1[currency] = 0.0
        dp1[initialCurrency] = 1.0

        for _ in range(num_currencies):
            updated = False
            for start_currency in list(dp1.keys()):
                if start_currency in day1_rates_map:
                    for target_currency, rate in day1_rates_map[start_currency].items():
                        if dp1[start_currency] * rate > dp1[target_currency]:
                            dp1[target_currency] = dp1[start_currency] * rate
                            updated = True
            if not updated:
                break

        dp2 = {}
        for currency in currencies:
            dp2[currency] = dp1[currency]

        for _ in range(num_currencies):
            updated = False
            for start_currency in list(dp2.keys()):
                if start_currency in day2_rates_map:
                    for target_currency, rate in day2_rates_map[start_currency].items():
                        if dp2[start_currency] * rate > dp2[target_currency]:
                            dp2[target_currency] = dp2[start_currency] * rate
                            updated = True
            if not updated:
                break

        return float(f'{dp2[initialCurrency]:.5f}')