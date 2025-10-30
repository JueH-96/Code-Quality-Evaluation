import collections
from typing import List, Dict, Set, Tuple

# Starter code structure provided by the user
class Solution:
    """
    Calculates the maximum amount of initialCurrency achievable after performing 
    currency conversions on day 1 and then day 2.
    """
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        
        # Helper function to build the conversion graph (adjacency list) 
        # and collect the set of unique currencies involved in the pairs.
        # The graph includes both forward (A->B) and inverse (B->A) conversions.
        def build_graph(pairs: List[List[str]], rates: List[float]) -> Tuple[Dict[str, List[Tuple[str, float]]], Set[str]]:
            """
            Builds an adjacency list representation of the currency conversion graph.

            Args:
                pairs: List of currency pairs [start, target].
                rates: List of corresponding conversion rates.

            Returns:
                A tuple containing:
                    - graph: A dictionary where keys are currency codes (str) and 
                             values are lists of tuples (neighbor_currency, rate).
                    - currencies: A set of unique currency codes involved.
            """
            graph = collections.defaultdict(list)
            currencies = set()
            for i, pair in enumerate(pairs):
                start, target = pair[0], pair[1]
                rate = rates[i]
                # Problem constraints state rates >= 1.0, ensuring rate is positive.
                graph[start].append((target, rate))
                # Add the inverse conversion with rate 1/rate
                graph[target].append((start, 1.0 / rate))
                # Keep track of all unique currencies mentioned in pairs
                currencies.add(start)
                currencies.add(target)
            return graph, currencies

        # Helper function using SPFA (Shortest Path Faster Algorithm) adapted for 
        # finding the maximum product path (maximum conversion rate).
        # Finds maximum rates FROM the given source currency TO all other reachable currencies.
        def run_spfa(source: str, graph: Dict[str, List[Tuple[str, float]]], currencies: Set[str]) -> Dict[str, float]:
            """
            Runs the SPFA algorithm to find maximum conversion rates from a source currency.

            Args:
                source: The starting currency code.
                graph: The conversion graph (adjacency list).
                currencies: The set of all known currencies.

            Returns:
                A dictionary mapping each currency code to the maximum amount achievable 
                starting with 1.0 of the source currency.
            """
            # Initialize maximum amounts reachable from the source. Default to 0.0.
            max_amounts = {c: 0.0 for c in currencies}
            
            # Ensure the source currency exists in our tracking, even if not in pairs.
            if source not in currencies:
                 currencies.add(source) 
                 max_amounts[source] = 0.0 # Initialize its entry

            # We start with 1.0 unit of the source currency.
            max_amounts[source] = 1.0
                 
            # Queue for the SPFA algorithm, initially containing only the source.
            queue = collections.deque([source])
            # Set to efficiently track which nodes are currently scheduled in the queue.
            in_queue = {c: False for c in currencies}
            in_queue[source] = True # Source starts in the queue

            # Safety counter to prevent extremely long runs or potential infinite loops 
            # (though problem constraints guarantee no arbitrage cycles within a single day).
            # Using V*V as a simple heuristic limit.
            process_limit = len(currencies) * len(currencies) 
            processed_count = 0

            while queue and processed_count < process_limit :
                u = queue.popleft() # Get the next currency to process
                in_queue[u] = False # Mark as no longer in the queue
                processed_count +=1

                # If the current currency 'u' has no outgoing edges defined in the graph, skip it.
                if u not in graph: 
                    continue

                # Explore conversions (relax edges) outgoing from 'u'.
                for v, rate in graph[u]:
                    # Ensure the target currency 'v' is tracked (should be if graph is built correctly).
                    if v not in max_amounts: 
                         # Add defensively if somehow missed.
                         currencies.add(v)
                         max_amounts[v] = 0.0
                         in_queue[v] = False # Ensure in_queue knows about 'v'

                    # Calculate the potential amount of 'v' achievable by converting from 'u'.
                    # Use .get(u, 0.0) for robustness in case u wasn't initialized properly (shouldn't happen here).
                    potential_amount = max_amounts.get(u, 0.0) * rate 
                    
                    # If this path (source -> ... -> u -> v) yields a higher amount for 'v'
                    # than previously known, update the maximum amount for 'v'.
                    # Using direct float comparison, as precision issues are unlikely given constraints.
                    if potential_amount > max_amounts.get(v, 0.0): # Use .get(v, 0.0) for safety
                        max_amounts[v] = potential_amount
                        # If 'v' was updated and is not already scheduled for processing, add it to the queue.
                        if not in_queue.get(v, False): 
                            queue.append(v)
                            in_queue[v] = True
            
            # Return the dictionary of maximum amounts reachable from the source.
            return max_amounts

        # Helper function using Floyd-Warshall algorithm adapted for finding the 
        # maximum product path between all pairs of currencies.
        def run_floyd_warshall_rates(graph: Dict[str, List[Tuple[str, float]]], currencies: Set[str]) -> Dict[str, Dict[str, float]]:
            """
            Runs the Floyd-Warshall algorithm to find all-pairs maximum conversion rates.

            Args:
                graph: The conversion graph (adjacency list).
                currencies: The set of all known currencies.

            Returns:
                A nested dictionary where result[start_currency][target_currency] gives 
                the maximum conversion rate.
            """
            # Create an ordered list for consistent indexing.
            currency_list = list(currencies)
            # Map currency codes to their index in the list.
            currency_to_idx = {c: i for i, c in enumerate(currency_list)}
            n = len(currency_list) # Number of unique currencies.
            
            # Initialize the matrix to store maximum rates. max_rates[i][j] = max rate from i to j.
            # Initialize all rates to 0.0.
            max_rates = [[0.0] * n for _ in range(n)]

            # Initialize base cases:
            # 1. Rate from a currency to itself is 1.0 (no conversion needed).
            # 2. Populate direct conversion rates from the input graph.
            for i in range(n):
                max_rates[i][i] = 1.0 # Rate from currency i to itself is 1.
                u = currency_list[i] # Get the currency code for index i.
                if u in graph: # Check if this currency has outgoing conversions.
                    for v_curr, rate in graph[u]: # For each direct conversion u -> v_curr
                        if v_curr in currency_to_idx: # Ensure target currency is in our list.
                            j = currency_to_idx[v_curr] # Get index of target currency.
                            # Update the direct rate, taking the max if multiple direct paths exist 
                            # (unlikely needed per problem statement, but safe).
                            max_rates[i][j] = max(max_rates[i][j], rate) 

            # Floyd-Warshall core logic: Iterate through all possible intermediate currencies 'k'.
            for k_idx in range(n):
                # Iterate through all possible starting currencies 'i'.
                for i_idx in range(n):
                    # If there's no path from 'i' to 'k' (rate is 0), we can't use 'k' as an intermediate 
                    # for paths starting at 'i', so skip the inner loop.
                    if max_rates[i_idx][k_idx] == 0.0: 
                        continue
                    # Iterate through all possible ending currencies 'j'.
                    for j_idx in range(n):
                        # If there's no path from 'k' to 'j' (rate is 0), we can't complete the path i->k->j.
                        if max_rates[k_idx][j_idx] == 0.0:
                             continue
                        
                        # Calculate the rate for the path i -> k -> j by multiplying rates.
                        potential_rate = max_rates[i_idx][k_idx] * max_rates[k_idx][j_idx]
                        
                        # If the rate via intermediate 'k' is better than the current best known rate from i to j, update it.
                        if potential_rate > max_rates[i_idx][j_idx]:
                            max_rates[i_idx][j_idx] = potential_rate

            # Convert the final rates matrix into a nested dictionary for convenient lookup by currency codes.
            result_dict = collections.defaultdict(lambda: collections.defaultdict(float))
            for i in range(n):
                start_curr = currency_list[i]
                for j in range(n):
                    target_curr = currency_list[j]
                     # Store only reachable pairs where the rate is greater than 0.
                    if max_rates[i][j] > 0.0:
                       result_dict[start_curr][target_curr] = max_rates[i][j]
            
            return result_dict


        # --- Main execution logic of maxAmount ---

        # 1. Build the conversion graphs for Day 1 and Day 2 using the provided pairs and rates.
        # Also, get the sets of currencies involved in each day's conversions.
        graph1, currencies1 = build_graph(pairs1, rates1)
        graph2, currencies2 = build_graph(pairs2, rates2)
        
        # Create a set containing all unique currencies involved across both days.
        all_currencies = currencies1.union(currencies2)
        # Crucially, ensure the initial currency is included in this set, even if it wasn't in any pairs.
        all_currencies.add(initialCurrency) 

        # 2. Calculate Day 1 results: Determine the maximum amount of *each* currency ('C') 
        # that can be obtained starting with 1.0 of `initialCurrency` after performing 
        # any number of conversions using Day 1 rates. Use SPFA for this single-source calculation.
        # Pass a copy of all_currencies set to avoid modification by the helper function.
        rates_day1 = run_spfa(initialCurrency, graph1, all_currencies.copy()) 

        # 3. Calculate Day 2 results: Determine the maximum conversion rates between *all pairs* 
        # of currencies using Day 2 rates. This is needed because after Day 1, we might hold 
        # any currency 'C', and we need to know the best rate to convert *from* 'C' *back to* 
        # `initialCurrency` on Day 2. Floyd-Warshall is suitable for all-pairs calculation.
        # Pass a copy of all_currencies set.
        all_pairs_rates_day2 = run_floyd_warshall_rates(graph2, all_currencies.copy())

        # 4. Combine Day 1 and Day 2 results: Find the overall maximum amount of `initialCurrency`.
        max_final_amount = 0.0

        # Iterate through every currency 'C' in our set of all currencies.
        # Consider 'C' as the potential currency held at the end of Day 1.
        for C in all_currencies:
            # Get the maximum amount of currency 'C' we could have after Day 1's conversions.
            # Use .get(C, 0.0) for safe access, defaulting to 0 if 'C' was unreachable.
            amount_C_after_day1 = rates_day1.get(C, 0.0) 
            
            # If currency 'C' was unreachable from initialCurrency on Day 1 (amount is 0), 
            # then this path through 'C' is impossible, so skip it.
            if amount_C_after_day1 == 0.0:
                continue

            # Get the maximum rate to convert from currency 'C' back to `initialCurrency` using Day 2 rates.
            # Access the nested dictionary safely using .get(). Defaults to 0.0 if the path C -> initialCurrency
            # doesn't exist on Day 2.
            rate_C_to_initial_day2 = all_pairs_rates_day2.get(C, {}).get(initialCurrency, 0.0)

            # If `initialCurrency` cannot be reached from 'C' on Day 2 (rate is 0), 
            # then this path cannot lead back to the target currency, so skip it.
            if rate_C_to_initial_day2 == 0.0:
                continue

            # Calculate the final amount of `initialCurrency` achieved by going through intermediate currency 'C'.
            # Final Amount = (Amount of C after Day 1) * (Rate C -> initialCurrency on Day 2)
            final_amount_via_C = amount_C_after_day1 * rate_C_to_initial_day2
            
            # Update the overall maximum amount found so far across all possible intermediate currencies 'C'.
            max_final_amount = max(max_final_amount, final_amount_via_C)

        # Return the highest amount of initialCurrency achievable.
        # Note: If no conversions are possible or beneficial, the loop iteration where C = initialCurrency 
        # will calculate amount_C_after_day1=1.0 (from SPFA init) * rate_C_to_initial_day2=1.0 (from FW init), 
        # resulting in a final amount of 1.0. So, the minimum returned value will be 1.0 if the initial
        # currency exists and isn't somehow made unreachable (which shouldn't happen).
        return max_final_amount