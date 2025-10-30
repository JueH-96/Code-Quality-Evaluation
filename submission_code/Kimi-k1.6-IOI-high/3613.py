from collections import defaultdict, deque
from typing import List

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        # Collect all unique currencies
        currencies = set()
        currencies.add(initialCurrency)
        for pair in pairs1 + pairs2:
            currencies.add(pair[0])
            currencies.add(pair[1])
        currencies = list(currencies)
        
        # Helper function to build the graph for a given set of pairs and rates
        def build_graph(pairs, rates):
            graph = defaultdict(list)
            for i in range(len(pairs)):
                start, end = pairs[i]
                rate = rates[i]
                graph[start].append((end, rate))
                graph[end].append((start, 1.0 / rate))
            return graph
        
        graph_day1 = build_graph(pairs1, rates1)
        graph_day2 = build_graph(pairs2, rates2)
        
        # Helper function to compute the maximum currency values using modified BFS (SPFA)
        def compute_max(start_currency, graph):
            max_amount = {curr: 0.0 for curr in currencies}
            max_amount[start_currency] = 1.0
            queue = deque([start_currency])
            in_queue = set([start_currency])
            
            while queue:
                curr = queue.popleft()
                in_queue.discard(curr)
                
                for neighbor, rate in graph[curr]:
                    if max_amount[curr] * rate > max_amount[neighbor]:
                        max_amount[neighbor] = max_amount[curr] * rate
                        if neighbor not in in_queue:
                            queue.append(neighbor)
                            in_queue.add(neighbor)
            
            return max_amount
        
        # Compute maximum amounts for day 1 starting from initialCurrency
        max_day1 = compute_max(initialCurrency, graph_day1)
        
        # Compute maximum conversion from each currency to initialCurrency using day 2's rates
        max_day2_to_initial = {}
        for curr in currencies:
            max_curr = compute_max(curr, graph_day2)
            max_day2_to_initial[curr] = max_curr.get(initialCurrency, 0.0)
        
        # Calculate the maximum product
        max_result = 0.0
        for curr in currencies:
            product = max_day1[curr] * max_day2_to_initial[curr]
            if product > max_result:
                max_result = product
        
        return max_result