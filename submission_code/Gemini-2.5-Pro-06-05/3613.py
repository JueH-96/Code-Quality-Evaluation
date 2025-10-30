from typing import List

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        
        # Helper function to compute all-pairs maximum exchange rates for a single day
        # using the Floyd-Warshall algorithm.
        def get_all_pairs_rates(pairs, rates, currency_map):
            num_currencies = len(currency_map)
            
            # dist[i][j] will store the max exchange rate from currency i to j.
            # Initialize with 0.0 for no path, and 1.0 for self-conversion.
            dist = [[0.0] * num_currencies for _ in range(num_currencies)]
            for i in range(num_currencies):
                dist[i][i] = 1.0

            # Populate the matrix with direct rates from the input pairs.
            # A conversion from u to v at rate r implies a conversion from v to u at rate 1/r.
            for i, (u, v) in enumerate(pairs):
                u_idx, v_idx = currency_map[u], currency_map[v]
                rate = rates[i]
                dist[u_idx][v_idx] = max(dist[u_idx][v_idx], rate)
                dist[v_idx][u_idx] = max(dist[v_idx][u_idx], 1.0 / rate)

            # Floyd-Warshall algorithm to find the best rates for all pairs.
            # This accounts for multi-step conversions.
            for k in range(num_currencies):
                for i in range(num_currencies):
                    for j in range(num_currencies):
                        # The best rate from i to j is either the current best,
                        # or the rate obtained by converting via an intermediate currency k.
                        rate_via_k = dist[i][k] * dist[k][j]
                        dist[i][j] = max(dist[i][j], rate_via_k)
            
            return dist

        # Step 1: Collect all unique currencies and map them to integer indices.
        # This allows us to use matrix-based graph algorithms.
        all_currencies = set([initialCurrency])
        for u, v in pairs1:
            all_currencies.add(u)
            all_currencies.add(v)
        for u, v in pairs2:
            all_currencies.add(u)
            all_currencies.add(v)
        
        # Sorting creates a deterministic mapping, which is good practice.
        currency_list = sorted(list(all_currencies))
        currency_map = {currency: i for i, currency in enumerate(currency_list)}

        # Step 2: Compute all-pairs best rates for Day 1.
        day1_rates_matrix = get_all_pairs_rates(pairs1, rates1, currency_map)
        
        # Step 3: Compute all-pairs best rates for Day 2.
        day2_rates_matrix = get_all_pairs_rates(pairs2, rates2, currency_map)
        
        # Step 4: Calculate the maximum final amount.
        # The overall strategy is to convert initialCurrency to some currency C on Day 1,
        # hold C, and then convert C back to initialCurrency on Day 2.
        # We iterate through all possible intermediate currencies C to find the optimal one.
        
        max_return = 0.0
        start_currency_idx = currency_map[initialCurrency]
        
        for i in range(len(currency_list)):
            intermediate_currency_idx = i
            
            # rate_day1 is the best rate to convert initialCurrency -> intermediate_currency on Day 1.
            # Starting with 1.0 initialCurrency, this is the amount of intermediate currency
            # we would have at the end of Day 1.
            rate_day1 = day1_rates_matrix[start_currency_idx][intermediate_currency_idx]
            
            # If the intermediate currency is unreachable on Day 1, this path is invalid.
            if rate_day1 == 0.0:
                continue
                
            # rate_day2 is the best rate to convert intermediate_currency -> initialCurrency on Day 2.
            rate_day2 = day2_rates_matrix[intermediate_currency_idx][start_currency_idx]
            
            # The total return for this strategy (starting with 1.0 of initialCurrency)
            # is the product of the two rates.
            current_return = rate_day1 * rate_day2
            max_return = max(max_return, current_return)
            
        return max_return