class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        # Get all unique currencies
        currencies = set()
        currencies.add(initialCurrency)
        for pair in pairs1:
            currencies.update(pair)
        for pair in pairs2:
            currencies.update(pair)
        
        currencies = list(currencies)
        n = len(currencies)
        currency_to_index = {currency: i for i, currency in enumerate(currencies)}
        
        # Initialize adjacency matrices for both days
        day1_graph = [[0.0] * n for _ in range(n)]
        day2_graph = [[0.0] * n for _ in range(n)]
        
        # Set diagonal to 1.0 (no conversion)
        for i in range(n):
            day1_graph[i][i] = 1.0
            day2_graph[i][i] = 1.0
        
        # Fill day1 graph
        for i, (start, target) in enumerate(pairs1):
            start_idx = currency_to_index[start]
            target_idx = currency_to_index[target]
            day1_graph[start_idx][target_idx] = max(day1_graph[start_idx][target_idx], rates1[i])
            day1_graph[target_idx][start_idx] = max(day1_graph[target_idx][start_idx], 1.0 / rates1[i])
        
        # Fill day2 graph
        for i, (start, target) in enumerate(pairs2):
            start_idx = currency_to_index[start]
            target_idx = currency_to_index[target]
            day2_graph[start_idx][target_idx] = max(day2_graph[start_idx][target_idx], rates2[i])
            day2_graph[target_idx][start_idx] = max(day2_graph[target_idx][start_idx], 1.0 / rates2[i])
        
        # Apply Floyd-Warshall for maximum rates
        def floyd_warshall_max(graph):
            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        if graph[i][k] > 0 and graph[k][j] > 0:
                            graph[i][j] = max(graph[i][j], graph[i][k] * graph[k][j])
        
        floyd_warshall_max(day1_graph)
        floyd_warshall_max(day2_graph)
        
        initial_idx = currency_to_index[initialCurrency]
        max_amount = 1.0  # No conversion case
        
        # Try all intermediate currencies
        for i in range(n):
            amount = day1_graph[initial_idx][i] * day2_graph[i][initial_idx]
            max_amount = max(max_amount, amount)
        
        return max_amount