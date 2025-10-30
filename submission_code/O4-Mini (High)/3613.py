from typing import List
from collections import defaultdict
import heapq

class Solution:
    def maxAmount(self, initialCurrency: str,
                  pairs1: List[List[str]], rates1: List[float],
                  pairs2: List[List[str]], rates2: List[float]) -> float:
        # Build bidirectional graphs for day1 and day2
        graph1 = defaultdict(list)
        for (u, v), r in zip(pairs1, rates1):
            graph1[u].append((v, r))
            graph1[v].append((u, 1.0 / r))
        graph2 = defaultdict(list)
        for (u, v), r in zip(pairs2, rates2):
            graph2[u].append((v, r))
            graph2[v].append((u, 1.0 / r))

        # Helper to compute, for a given start currency, the maximum product
        # conversion to every other currency in the given graph.
        def max_prod(start: str, graph: defaultdict) -> dict:
            best = {start: 1.0}
            # Use a max-heap (by pushing negative product) to explore best paths first
            heap = [(-1.0, start)]
            while heap:
                neg_prod, u = heapq.heappop(heap)
                curr_prod = -neg_prod
                # stale entry?
                if curr_prod < best[u] - 1e-12:
                    continue
                for v, rate in graph.get(u, []):
                    new_prod = curr_prod * rate
                    if new_prod > best.get(v, 0.0) + 1e-12:
                        best[v] = new_prod
                        heapq.heappush(heap, (-new_prod, v))
            return best

        # f1[x] = max amount of currency x we can get on day1 starting from 1 initialCurrency
        f1 = max_prod(initialCurrency, graph1)

        # Now for each currency x reachable on day1, see how much initialCurrency
        # we can end up with by converting x back to initialCurrency on day2
        ans = 0.0
        for curr, amount_day1 in f1.items():
            f2 = max_prod(curr, graph2)
            amount_final = amount_day1 * f2.get(initialCurrency, 0.0)
            ans = max(ans, amount_final)

        return ans