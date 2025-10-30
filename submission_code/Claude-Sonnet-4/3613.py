class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        from collections import defaultdict, deque
        
        def build_graph(pairs, rates):
            graph = defaultdict(list)
            for i, (start, target) in enumerate(pairs):
                rate = rates[i]
                graph[start].append((target, rate))
                graph[target].append((start, 1.0 / rate))
            return graph
        
        def get_max_rates(graph, start_currency):
            # BFS to find maximum conversion rates from start_currency to all other currencies
            max_rates = {start_currency: 1.0}
            queue = deque([(start_currency, 1.0)])
            
            while queue:
                curr_currency, curr_rate = queue.popleft()
                
                for next_currency, rate in graph[curr_currency]:
                    new_rate = curr_rate * rate
                    if next_currency not in max_rates or new_rate > max_rates[next_currency]:
                        max_rates[next_currency] = new_rate
                        queue.append((next_currency, new_rate))
            
            return max_rates
        
        # Build graphs for both days
        graph1 = build_graph(pairs1, rates1)
        graph2 = build_graph(pairs2, rates2)
        
        # Get maximum rates from initial currency on day 1
        day1_rates = get_max_rates(graph1, initialCurrency)
        
        # Get maximum rates to initial currency on day 2
        day2_rates = get_max_rates(graph2, initialCurrency)
        
        # Find maximum amount possible
        max_amount = 1.0  # Starting with 1.0 of initial currency
        
        # Try all possible intermediate currencies
        all_currencies = set()
        for pair in pairs1:
            all_currencies.update(pair)
        for pair in pairs2:
            all_currencies.update(pair)
        all_currencies.add(initialCurrency)
        
        for intermediate_currency in all_currencies:
            # Rate to convert from initial to intermediate on day 1
            day1_rate = day1_rates.get(intermediate_currency, 0)
            # Rate to convert from intermediate back to initial on day 2
            day2_rate = day2_rates.get(intermediate_currency, 0)
            
            if day1_rate > 0 and day2_rate > 0:
                total_rate = day1_rate * day2_rate
                max_amount = max(max_amount, total_rate)
        
        return max_amount