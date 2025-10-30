from typing import List, Dict
from collections import defaultdict, deque

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        def build_graph(pairs: List[List[str]], rates: List[float]) -> Dict[str, Dict[str, float]]:
            graph = defaultdict(dict)
            for (start, target), rate in zip(pairs, rates):
                graph[start][target] = rate
                graph[target][start] = 1 / rate
            return graph
        
        def bellman_ford(graph: Dict[str, Dict[str, float]], start: str) -> Dict[str, float]:
            max_amount = defaultdict(lambda: float('-inf'))
            max_amount[start] = 1.0
            queue = deque([start])
            in_queue = set([start])
            
            while queue:
                current = queue.popleft()
                in_queue.remove(current)
                
                for neighbor, rate in graph[current].items():
                    if max_amount[current] * rate > max_amount[neighbor]:
                        max_amount[neighbor] = max_amount[current] * rate
                        if neighbor not in in_queue:
                            queue.append(neighbor)
                            in_queue.add(neighbor)
            
            return max_amount
        
        # Build graphs for both days
        graph_day1 = build_graph(pairs1, rates1)
        graph_day2 = build_graph(pairs2, rates2)
        
        # Calculate maximum amounts after day 1
        max_after_day1 = bellman_ford(graph_day1, initialCurrency)
        
        # Calculate maximum amounts after day 2 starting from each currency obtained after day 1
        max_final_amount = 0.0
        for currency, amount in max_after_day1.items():
            if amount > 0:
                max_after_day2 = bellman_ford(graph_day2, currency)
                max_final_amount = max(max_final_amount, max_after_day2[initialCurrency] * amount)
        
        return max_final_amount