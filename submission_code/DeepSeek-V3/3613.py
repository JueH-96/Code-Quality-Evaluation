from collections import defaultdict

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        # Build graph for day 1
        graph1 = defaultdict(dict)
        for (start, target), rate in zip(pairs1, rates1):
            graph1[start][target] = rate
            graph1[target][start] = 1.0 / rate
        
        # Build graph for day 2
        graph2 = defaultdict(dict)
        for (start, target), rate in zip(pairs2, rates2):
            graph2[start][target] = rate
            graph2[target][start] = 1.0 / rate
        
        # Function to compute the maximum amount for a given graph and starting currency
        def get_max_amount(graph, start_currency):
            amounts = defaultdict(float)
            amounts[start_currency] = 1.0
            queue = [start_currency]
            while queue:
                current = queue.pop(0)
                for neighbor, rate in graph[current].items():
                    new_amount = amounts[current] * rate
                    if new_amount > amounts[neighbor]:
                        amounts[neighbor] = new_amount
                        queue.append(neighbor)
            return amounts
        
        # Get all possible amounts after day 1
        day1_amounts = get_max_amount(graph1, initialCurrency)
        
        # For each currency after day 1, compute the maximum amount after day 2
        max_final_amount = 0.0
        for currency, amount in day1_amounts.items():
            day2_amounts = get_max_amount(graph2, currency)
            if initialCurrency in day2_amounts:
                final_amount = amount * day2_amounts[initialCurrency]
                if final_amount > max_final_amount:
                    max_final_amount = final_amount
        
        return max_final_amount