from typing import List

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        def process_day(initial_amounts, pairs, rates):
            current_amounts = initial_amounts.copy()
            conversion_rates = {}
            currencies = set(current_amounts.keys())
            for i in range(len(pairs)):
                start_currency, target_currency = pairs[i]
                rate = rates[i]
                conversion_rates[(start_currency, target_currency)] = rate
                conversion_rates[(target_currency, start_currency)] = 1.0 / rate
                currencies.add(start_currency)
                currencies.add(target_currency)

            for currency in currencies:
                if currency not in current_amounts:
                    current_amounts[currency] = 0.0

            updated = True
            while updated:
                updated = False
                for start_currency in list(current_amounts.keys()):
                    if start_currency not in current_amounts:
                        continue
                    for target_currency in currencies:
                        if (start_currency, target_currency) in conversion_rates:
                            rate = conversion_rates[(start_currency, target_currency)]
                            new_amount = current_amounts[start_currency] * rate
                            if target_currency not in current_amounts or new_amount > current_amounts[target_currency]:
                                current_amounts[target_currency] = new_amount
                                updated = True
            return current_amounts

        amounts_day0 = {initialCurrency: 1.0}
        amounts_day1 = process_day(amounts_day0, pairs1, rates1)
        amounts_day2 = process_day(amounts_day1, pairs2, rates2)
        return amounts_day2.get(initialCurrency, 1.0)