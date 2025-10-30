class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:

        def calculate_max_amounts(initial_amounts, pairs, rates):
            amounts = initial_amounts.copy()
            while True:
                changed = False
                for (start, target), rate in zip(pairs, rates):
                    if start in amounts:
                        new_amount = amounts[start] * rate
                        if target not in amounts or new_amount > amounts.get(target, 0.0):
                            amounts[target] = new_amount
                            changed = True

                    if target in amounts:
                        reverse_rate = 1.0 / rate
                        new_amount = amounts[target] * reverse_rate
                        if start not in amounts or new_amount > amounts.get(start, 0.0):
                            amounts[start] = new_amount
                            changed = True
                if not changed:
                    break
            return amounts

        # Day 1
        amounts1 = {initialCurrency: 1.0}
        amounts1 = calculate_max_amounts(amounts1, pairs1, rates1)

        # Day 2
        amounts2 = amounts1.copy()
        amounts2 = calculate_max_amounts(amounts2, pairs2, rates2)

        return amounts2.get(initialCurrency, 1.0)