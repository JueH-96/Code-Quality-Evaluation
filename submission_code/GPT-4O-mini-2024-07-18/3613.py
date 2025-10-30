from typing import List
from collections import defaultdict

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        def build_graph(pairs, rates):
            graph = defaultdict(dict)
            for (start, target), rate in zip(pairs, rates):
                graph[start][target] = rate
            return graph
        
        def max_conversion(graph, currency):
            # Use DFS to find the maximum conversion rate for a given currency
            max_amount = 1.0
            stack = [(currency, 1.0)]
            visited = set()
            
            while stack:
                current_currency, current_amount = stack.pop()
                if current_currency in visited:
                    continue
                visited.add(current_currency)
                max_amount = max(max_amount, current_amount)
                
                for neighbor, rate in graph[current_currency].items():
                    stack.append((neighbor, current_amount * rate))
                    
            return max_amount
        
        # Build conversion graphs for both days
        graph1 = build_graph(pairs1, rates1)
        graph2 = build_graph(pairs2, rates2)
        
        # Calculate maximum amount after day 1
        max_after_day1 = max_conversion(graph1, initialCurrency)
        
        # Calculate maximum amount after day 2 starting from all currencies we can reach from initialCurrency
        max_final_amount = 0.0
        
        # We can convert back to initialCurrency using inverse rates
        for currency in graph1[initialCurrency]:
            amount_after_day1 = max_after_day1 * graph1[initialCurrency][currency]
            # Now we can convert this amount on day 2
            max_after_day2 = max_conversion(graph2, currency)
            max_final_amount = max(max_final_amount, amount_after_day1 * max_after_day2)
        
        # Also consider the case where we do not convert on day 1
        max_final_amount = max(max_final_amount, max_after_day1)
        
        return max_final_amount