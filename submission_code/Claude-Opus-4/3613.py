class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        from collections import defaultdict
        
        # Build graphs for both days
        def build_graph(pairs, rates):
            graph = defaultdict(list)
            for i in range(len(pairs)):
                start, end = pairs[i]
                rate = rates[i]
                # Add forward conversion
                graph[start].append((end, rate))
                # Add reverse conversion
                graph[end].append((start, 1.0 / rate))
            return graph
        
        # Find maximum rates from a starting currency to all reachable currencies
        def find_max_rates(graph, start_currency):
            max_rates = {start_currency: 1.0}
            # Use a queue for BFS-like traversal
            from collections import deque
            queue = deque([(start_currency, 1.0)])
            
            # Keep processing until we can't improve any rates
            while queue:
                currency, current_rate = queue.popleft()
                
                # Try all conversions from current currency
                for next_currency, conversion_rate in graph[currency]:
                    new_rate = current_rate * conversion_rate
                    
                    # If we found a better rate to next_currency, update it
                    if next_currency not in max_rates or new_rate > max_rates[next_currency]:
                        max_rates[next_currency] = new_rate
                        queue.append((next_currency, new_rate))
            
            return max_rates
        
        # Build graphs for both days
        graph1 = build_graph(pairs1, rates1)
        graph2 = build_graph(pairs2, rates2)
        
        # Find max rates from initialCurrency on day 1
        day1_rates = find_max_rates(graph1, initialCurrency)
        
        # For each currency reachable on day 1, find the best path back to initialCurrency on day 2
        max_amount = 1.0  # Starting with 1.0 of initialCurrency
        
        for intermediate_currency, day1_rate in day1_rates.items():
            # Find max rates from intermediate_currency on day 2
            day2_rates = find_max_rates(graph2, intermediate_currency)
            
            # Check if we can get back to initialCurrency
            if initialCurrency in day2_rates:
                total_rate = day1_rate * day2_rates[initialCurrency]
                max_amount = max(max_amount, total_rate)
        
        return max_amount