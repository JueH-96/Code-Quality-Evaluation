from typing import List
from collections import defaultdict, deque

class Solution:
    def maxAmount(self, initialCurrency: str, 
                  pairs1: List[List[str]], rates1: List[float],
                  pairs2: List[List[str]], rates2: List[float]) -> float:
        
        # ---- helper to build an undirected (bidirectional) graph -------------
        # every quoted rate is stored in the forward direction,
        # and its reciprocal in the opposite direction
        def build_graph(pairs: List[List[str]], rates: List[float]):
            g = defaultdict(list)
            for (a, b), r in zip(pairs, rates):
                g[a].append((b, r))       # a  -> b  with  r
                g[b].append((a, 1.0 / r)) # b  -> a  with 1/r
            return g
        
        # ---- helper to compute all rates from `initialCurrency` --------------
        # because the graph is guaranteed to be acyclic / contradiction-free,
        # the product of edge-rates along the unique path is well defined
        def rates_from_initial(graph):
            rates = {}                               # currency -> rate
            stack = [(initialCurrency, 1.0)]
            while stack:
                cur, val = stack.pop()
                if cur in rates:
                    continue
                rates[cur] = val
                for nxt, r in graph[cur]:
                    if nxt not in rates:
                        stack.append((nxt, val * r))
            return rates
        
        # build the two daily graphs
        g1 = build_graph(pairs1, rates1)
        g2 = build_graph(pairs2, rates2)
        
        # rate from initialCurrency to every reachable currency for each day
        day1_rate = rates_from_initial(g1)           # initial -> X  (day 1)
        day2_rate = rates_from_initial(g2)           # initial -> X  (day 2)
        # Note: rate to go *back* on day 2 is simply the reciprocal:
        # X -> initial  ==  1 / (initial -> X)
        
        # try ending day 1 with every currency X that is reachable
        # on both days and pick the best product
        best = 1.0                                   # doing nothing
        for cur, r1 in day1_rate.items():
            if cur in day2_rate:                     # must be reachable on day 2
                gain = r1 / day2_rate[cur]           # r1 * (1/day2_rate)
                best = max(best, gain)
        
        return best