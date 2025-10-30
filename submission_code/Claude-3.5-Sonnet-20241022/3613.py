class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        # Build conversion graphs for both days
        def build_graph(pairs, rates):
            graph = {}
            for (start, target), rate in zip(pairs, rates):
                if start not in graph:
                    graph[start] = {}
                if target not in graph:
                    graph[target] = {}
                graph[start][target] = rate
                graph[target][start] = 1.0 / rate
            return graph
        
        graph1 = build_graph(pairs1, rates1)
        graph2 = build_graph(pairs2, rates2)
        
        # Get all possible amounts after day 1
        def get_amounts(curr, amount, graph, visited):
            results = {curr: amount}  # Store currency and amount
            visited.add(curr)
            
            if curr in graph:
                for next_curr, rate in graph[curr].items():
                    if next_curr not in visited:
                        next_amounts = get_amounts(next_curr, amount * rate, graph, visited)
                        results.update(next_amounts)
            
            visited.remove(curr)
            return results
        
        # Get all possible amounts after day 1
        day1_amounts = get_amounts(initialCurrency, 1.0, graph1, set())
        
        # Try all possible conversions on day 2 starting from each amount from day 1
        max_amount = 1.0  # Initialize with starting amount
        
        for curr, amount in day1_amounts.items():
            day2_amounts = get_amounts(curr, amount, graph2, set())
            
            # Find maximum amount in initialCurrency
            for final_curr, final_amount in day2_amounts.items():
                if final_curr == initialCurrency:
                    max_amount = max(max_amount, final_amount)
        
        return max_amount