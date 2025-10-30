from typing import List, Dict, Set

class Solution:
    def _compute_all_pairs_rates(self, currencies_list: List[str], pairs: List[List[str]], rates_values: List[float]) -> Dict[str, Dict[str, float]]:
        num_curr = len(currencies_list)
        
        # If there are no currencies (e.g., if initialCurrency was empty and no pairs,
        # though problem constraints prevent this), return an empty map.
        if num_curr == 0:
            return {}
            
        curr_to_idx = {curr: i for i, curr in enumerate(currencies_list)}
        
        # adj[i][j] stores the maximum rate from currency i to currency j
        # Initialize with 0.0, meaning no conversion path known yet.
        adj = [[0.0] * num_curr for _ in range(num_curr)]
        
        # Rate from a currency to itself is 1.0 (by doing nothing)
        for i in range(num_curr):
            adj[i][i] = 1.0
            
        # Populate initial direct conversion rates from input pairs
        for k_pair_idx in range(len(pairs)):
            u_str, v_str = pairs[k_pair_idx]
            rate = rates_values[k_pair_idx]
            
            # All currencies in pairs should have been added to currencies_list,
            # so they must exist in curr_to_idx.
            u_idx, v_idx = curr_to_idx[u_str], curr_to_idx[v_str]
            
            # Use max in case a direct pair is defined multiple times with different rates,
            # or if a better rate for this direct pair was already found via another path
            # (though Floyd-Warshall handles indirect paths later).
            # Problem statement "no contradictions" implies this might not be strictly necessary for direct rates.
            adj[u_idx][v_idx] = max(adj[u_idx][v_idx], rate)
            # Inverse rate for V -> U. Constraints: rates_values[i] >= 1.0, so rate is positive.
            adj[v_idx][u_idx] = max(adj[v_idx][u_idx], 1.0 / rate)
            
        # Floyd-Warshall algorithm to find all-pairs best rates
        for k_idx in range(num_curr):      # Intermediate currency index
            for i_idx in range(num_curr):  # Start currency index
                for j_idx in range(num_curr):  # Target currency index
                    # If a path from currency i to k, and from k to j exists
                    if adj[i_idx][k_idx] > 0.0 and adj[k_idx][j_idx] > 0.0:
                        new_rate = adj[i_idx][k_idx] * adj[k_idx][j_idx]
                        # If this path (i -> k -> j) yields a better rate than previously known for i -> j
                        adj[i_idx][j_idx] = max(adj[i_idx][j_idx], new_rate)
                        
        # Convert the adjacency matrix to a dictionary of dictionaries for easier lookup by currency string
        rates_map_ret: Dict[str, Dict[str, float]] = {curr: {} for curr in currencies_list}
        for i in range(num_curr):
            curr_i_str = currencies_list[i]
            for j in range(num_curr):
                curr_j_str = currencies_list[j]
                rates_map_ret[curr_i_str][curr_j_str] = adj[i][j]
                
        return rates_map_ret

    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        # Step 1: Identify all unique currencies
        all_curr_set: Set[str] = set()
        all_curr_set.add(initialCurrency) # initialCurrency.length >= 1 constraint
        
        for p in pairs1:
            all_curr_set.add(p[0])
            all_curr_set.add(p[1])
        for p in pairs2:
            all_curr_set.add(p[0])
            all_curr_set.add(p[1])
        
        currencies_list = list(all_curr_set) # Guarantees initialCurrency is in list
        
        # Step 2: Compute all-pairs effective rates for Day 1
        day1_rates_map = self._compute_all_pairs_rates(currencies_list, pairs1, rates1)
        
        # Step 3: Compute all-pairs effective rates for Day 2
        day2_rates_map = self._compute_all_pairs_rates(currencies_list, pairs2, rates2)
        
        # Step 4: Determine the maximum final amount
        max_final_amount = 0.0
        
        # initialCurrency is guaranteed to be in currencies_list and thus a key in day1_rates_map.
        initial_curr_day1_options = day1_rates_map[initialCurrency]

        for interm_curr_str in currencies_list:
            # Rate to convert 1.0 initialCurrency to interm_curr_str on Day 1
            # This is effectively the amount of interm_curr_str we'd have.
            rate_initial_to_interm_day1 = initial_curr_day1_options.get(interm_curr_str, 0.0)
            
            if rate_initial_to_interm_day1 == 0.0:
                # Cannot reach interm_curr_str from initialCurrency on Day 1
                continue
            
            # Rate to convert 1.0 interm_curr_str back to initialCurrency on Day 2
            # interm_curr_str is guaranteed to be in currencies_list and thus a key in day2_rates_map.
            interm_curr_day2_options = day2_rates_map[interm_curr_str]
            rate_interm_to_initial_day2 = interm_curr_day2_options.get(initialCurrency, 0.0)
            
            if rate_interm_to_initial_day2 == 0.0:
                # Cannot convert interm_curr_str back to initialCurrency on Day 2
                continue
                
            # Calculate final amount of initialCurrency for this intermediate currency path
            # Start with 1.0 of initialCurrency.
            # After Day 1: 1.0 * rate_initial_to_interm_day1 units of interm_curr_str.
            # After Day 2: (1.0 * rate_initial_to_interm_day1) * rate_interm_to_initial_day2 units of initialCurrency.
            current_total_amount = rate_initial_to_interm_day1 * rate_interm_to_initial_day2
            max_final_amount = max(max_final_amount, current_total_amount)
            
        return max_final_amount