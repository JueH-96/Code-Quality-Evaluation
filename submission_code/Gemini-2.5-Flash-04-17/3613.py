from typing import List

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        # 1. Identify all unique currencies. Map them to indices.
        all_currencies = set([initialCurrency])
        for pair in pairs1:
            all_currencies.add(pair[0])
            all_currencies.add(pair[1])
        for pair in pairs2:
            all_currencies.add(pair[0])
            all_currencies.add(pair[1])

        currency_list = list(all_currencies)
        n_currencies = len(currency_list)
        currency_to_idx = {c: i for i, c in enumerate(currency_list)}

        # 2. Build the list of Day 1 edges (forward and inverse).
        # Store original pairs and rates for easier iteration in relaxation step.
        day1_pairs_rates = []
        for i in range(len(pairs1)):
            u, v = pairs1[i]
            rate = rates1[i]
            u_idx = currency_to_idx[u]
            v_idx = currency_to_idx[v]
            day1_pairs_rates.append((u_idx, v_idx, rate))

        # 3. Compute max_amount_day1 using Bellman-Ford-like maximization.
        # max_amount_day1[i] = max amount of currency_list[i] obtainable from 1 unit of initialCurrency on Day 1.
        max_amount_day1 = [0.0] * n_currencies
        initial_currency_idx = currency_to_idx[initialCurrency]
        max_amount_day1[initial_currency_idx] = 1.0

        # Run relaxation n_currencies times (sufficient for convergence without positive cycles)
        for _ in range(n_currencies):
             # Iterate through original pairs to apply both forward and inverse edge relaxations
             for u_idx, v_idx, rate in day1_pairs_rates:
                 # Relax edge u -> v
                 # Max amount at v can be improved by coming from u
                 if max_amount_day1[u_idx] > 0: # Optimization: avoid propagating from unreachable nodes initially
                     max_amount_day1[v_idx] = max(max_amount_day1[v_idx], max_amount_day1[u_idx] * rate)

                 # Relax edge v -> u (using inverse rate)
                 # Max amount at u can be improved by coming from v
                 if max_amount_day1[v_idx] > 0: # Optimization
                     max_amount_day1[u_idx] = max(max_amount_day1[u_idx], max_amount_day1[v_idx] * (1.0 / rate))

        # Note: The > 0 optimization is not strictly necessary as max(x, 0) = x if x >= 0, but can slightly prune updates.
        # The standard Bellman-Ford loop without the `if > 0` check would also work.
        # Let's use the standard version for robustness.
        max_amount_day1 = [0.0] * n_currencies
        max_amount_day1[initial_currency_idx] = 1.0
        for _ in range(n_currencies):
             for u_idx, v_idx, rate in day1_pairs_rates:
                 max_amount_day1[v_idx] = max(max_amount_day1[v_idx], max_amount_day1[u_idx] * rate)
                 max_amount_day1[u_idx] = max(max_amount_day1[u_idx], max_amount_day1[v_idx] * (1.0 / rate))


        # 4. Build the list of Day 2 pairs and rates.
        day2_pairs_rates = []
        for i in range(len(pairs2)):
             u, v = pairs2[i]
             rate = rates2[i]
             u_idx = currency_to_idx[u]
             v_idx = currency_to_idx[v]
             day2_pairs_rates.append((u_idx, v_idx, rate))

        # 5. Compute max_factor_day2_from using Bellman-Ford-like maximization.
        # max_factor_day2_from[i] = max factor to convert 1 unit of currency_list[i] to initialCurrency on Day 2.
        # This is effectively running Bellman-Ford on the reverse graph to find the max path from initialCurrency.
        max_factor_day2_from = [0.0] * n_currencies
        max_factor_day2_from[initial_currency_idx] = 1.0 # Base case: 1 unit of initialCurrency converts to 1 unit of initialCurrency

        # Run relaxation n_currencies times
        for _ in range(n_currencies):
            # Iterate through original pairs and rates for Day 2
            # For an edge u -> v with rate r:
            # Max factor from u to initialCurrency can be improved via path u -> v -> ... -> initialCurrency. Factor = r * max_factor_day2_from[v].
            # Max factor from v to initialCurrency can be improved via path v -> u -> ... -> initialCurrency. Factor = (1/r) * max_factor_day2_from[u].
            for u_idx, v_idx, rate in day2_pairs_rates:
                 # Update factor FROM u_idx (using edge u_idx -> v_idx)
                 max_factor_day2_from[u_idx] = max(max_factor_day2_from[u_idx], rate * max_factor_day2_from[v_idx])

                 # Update factor FROM v_idx (using edge v_idx -> u_idx with inverse rate)
                 max_factor_day2_from[v_idx] = max(max_factor_day2_from[v_idx], (1.0 / rate) * max_factor_day2_from[u_idx])


        # 6. Compute the final result.
        # The final amount is the maximum amount obtainable by converting all wealth
        # to some currency 'c' at the end of Day 1, and then converting that
        # amount from 'c' back to initialCurrency on Day 2.
        # This means finding max(amount_of_c_day1 * factor_from_c_day2_to_initial for all c)
        max_final_amount = 0.0
        for i in range(n_currencies):
            max_final_amount = max(max_final_amount, max_amount_day1[i] * max_factor_day2_from[i])

        return max_final_amount