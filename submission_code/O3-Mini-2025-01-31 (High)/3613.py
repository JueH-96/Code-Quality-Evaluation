from collections import defaultdict, deque
from typing import List

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], 
                  pairs2: List[List[str]], rates2: List[float]) -> float:
        # Helper function: Given a graph that maps currency -> list of (neighbor, rate),
        # compute maximum conversion factor from source to every other currency.
        def compute_max_conversion(graph: dict, source: str) -> dict:
            best = defaultdict(float)
            best[source] = 1.0
            dq = deque([source])
            while dq:
                curr = dq.popleft()
                curr_val = best[curr]
                for neighbor, rate in graph.get(curr, []):
                    new_val = curr_val * rate
                    # update if we get a better conversion factor
                    if new_val > best[neighbor]:
                        best[neighbor] = new_val
                        dq.append(neighbor)
            return best

        # Build graph for Day 1.
        # For each pair, we add both the given conversion and its inverse.
        graph1 = defaultdict(list)
        for (cur_from, cur_to), rate in zip(pairs1, rates1):
            graph1[cur_from].append((cur_to, rate))
            graph1[cur_to].append((cur_from, 1.0 / rate))
        
        # Compute the best conversion multipliers on Day 1 from initialCurrency.
        best_day1 = compute_max_conversion(graph1, initialCurrency)
        
        # Build graph for Day 2.
        graph2 = defaultdict(list)
        for (cur_from, cur_to), rate in zip(pairs2, rates2):
            graph2[cur_from].append((cur_to, rate))
            graph2[cur_to].append((cur_from, 1.0 / rate))
        
        # To get the maximum conversion from any currency x to the initialCurrency on Day 2,
        # we construct the reversed graph of graph2.
        rev_graph2 = defaultdict(list)
        for u in graph2:
            for v, rate in graph2[u]:
                # Reverse the edge: from v to u with the same multiplier.
                rev_graph2[v].append((u, rate))
        
        # Now, compute maximum conversion multiplier from initialCurrency on the reversed Day2 graph.
        # For any currency x, this value equals the maximum product for a conversion from x to initialCurrency in the original Day2 graph.
        best_rev_day2 = compute_max_conversion(rev_graph2, initialCurrency)
        
        # Evaluate the best final amount in initialCurrency by combining Day 1 and Day 2 conversions.
        # For each currency x reachable on Day 1 where we can later convert back to initialCurrency on Day 2,
        # the total conversion factor is best_day1[x] * best_rev_day2[x].
        max_final = 0.0
        for currency, factor_day1 in best_day1.items():
            if currency in best_rev_day2:
                candidate = factor_day1 * best_rev_day2[currency]
                if candidate > max_final:
                    max_final = candidate
        
        return max_final