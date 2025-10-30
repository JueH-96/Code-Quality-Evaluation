import math
from typing import List

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        # Find all unique currencies
        all_currencies = set([initialCurrency])
        for pair in pairs1:
            all_currencies.add(pair[0])
            all_currencies.add(pair[1])
        for pair in pairs2:
            all_currencies.add(pair[0])
            all_currencies.add(pair[1])
        
        # Build edges for day1 with w_sp weights
        edges_day1_wsp = []
        for i in range(len(pairs1)):
            A, B = pairs1[i]
            rate_AB = rates1[i]
            edges_day1_wsp.append((A, B, -math.log(rate_AB)))  # A to B
            edges_day1_wsp.append((B, A, math.log(rate_AB)))   # B to A with w_sp = log(rate_AB)
        
        # Build edges for day2 with w_sp weights
        edges_day2_wsp = []
        for i in range(len(pairs2)):
            A, B = pairs2[i]
            rate_AB = rates2[i]
            edges_day2_wsp.append((A, B, -math.log(rate_AB)))  # A to B
            edges_day2_wsp.append((B, A, math.log(rate_AB)))   # B to A
        
        # Define Bellman-Ford function
        def bellman_ford(nodes, edges, source):
            dist = {node: float('inf') for node in nodes}
            dist[source] = 0.0
            num_nodes = len(nodes)
            for _ in range(num_nodes - 1):
                for frm, to, weight in edges:
                    if dist[frm] < float('inf'):
                        new_dist = dist[frm] + weight
                        if new_dist < dist[to]:
                            dist[to] = new_dist
            return dist
        
        # Run Bellman-Ford for day1 from initialCurrency
        dist1 = bellman_ford(all_currencies, edges_day1_wsp, initialCurrency)
        
        # Build reversed edges for day2
        reversed_edges_day2_wsp = [(to, frm, weight) for frm, to, weight in edges_day2_wsp]
        
        # Run Bellman-Ford on reversed day2 graph from initialCurrency
        dist_to_initial_day2 = bellman_ford(all_currencies, reversed_edges_day2_wsp, initialCurrency)
        
        # Now find the maximum multiplier
        max_mult = 0.0  # Initialize to 0, will be updated to at least 1.0
        for C in all_currencies:
            if dist1[C] < float('inf') and dist_to_initial_day2[C] < float('inf'):
                sum_dist = dist1[C] + dist_to_initial_day2[C]
                mult = math.exp(-sum_dist)
                if mult > max_mult:
                    max_mult = mult
        
        return max_mult