from typing import List
import collections

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], 
                  pairs2: List[List[str]], rates2: List[float]) -> float:
        # Build directed edges with weights for a given day's pairs and rates
        def build_edges(pairs: List[List[str]], rates: List[float]):
            edges = []
            for (u, v), r in zip(pairs, rates):
                edges.append((u, v, r))
                edges.append((v, u, 1.0 / r))
            return edges

        # Given initial amounts for certain currencies, and a set of edges (u->v @ rate),
        # relax until no improvement: amounts[v] = max(amounts[v], amounts[u] * rate)
        def relax(amounts: dict, edges: List[tuple]):
            q = collections.deque(amounts.keys())
            # We'll keep a set to know who's in queue (optional optimization)
            in_queue = set(q)
            while q:
                u = q.popleft()
                in_queue.remove(u)
                amt_u = amounts[u]
                for (x, y, r) in edges:
                    if x == u:
                        cand = amt_u * r
                        if cand > amounts.get(y, 0.0):
                            amounts[y] = cand
                            if y not in in_queue:
                                q.append(y)
                                in_queue.add(y)
            return amounts

        # Day 1: start with 1.0 of initialCurrency
        edges1 = build_edges(pairs1, rates1)
        amounts_day1 = { initialCurrency: 1.0 }
        amounts_day1 = relax(amounts_day1, edges1)

        # Day 2: start with whatever we have at end of day1 in each currency
        edges2 = build_edges(pairs2, rates2)
        # copy the day1 amounts as the "initial" amounts on day2
        amounts_day2 = dict(amounts_day1)
        amounts_day2 = relax(amounts_day2, edges2)

        # The answer is how much initialCurrency we can end with
        return amounts_day2.get(initialCurrency, 1.0)