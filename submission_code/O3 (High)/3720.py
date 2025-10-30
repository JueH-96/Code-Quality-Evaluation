from typing import List
import heapq

class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        """
        Idea
        ----
        A node `u` can reach node `0` in the original graph **iff**
        node `0` can reach `u` in the graph where every edge is reversed.
        For every node we would like to know the smallest possible
        value of the largest edge–weight that has to be used on a path
        leading to `0`.  
        This is the classical *min-max / bottleneck* shortest–path problem
        and can be solved with a Dijkstra-like algorithm where the
        cost of a path is `max(weight on the path)`.

        Let  
            dist[u] = smallest possible maximal weight on any path
                       (following the original directions) from u to 0
        and let W = max_u dist[u].

        •  If a node is unreachable, dist[u] = ∞  ⇒  impossible (return -1).  
        •  Otherwise W is the minimal achievable value for the
           maximum edge weight: with every node we take one
           edge of a dist-minimal path that leads *closer* to the root,
           producing an arborescence where each node has out-degree 1
           (≤ threshold because threshold ≥ 1 by constraints).

        Algorithm
        ---------
        1. Build the reversed graph: for every (a, b, w) add (b → a, w).
        2. Run bottleneck-Dijkstra from node 0 on that reversed graph:
             next_cost = max(curr_cost, w)
        3. If some node was not reached ⇒ return -1.
        4. Answer = max(dist).

        Complexity
        ----------
        Time :  O((n + |E|) · log n)   (|E| ≤ 1e5)  
        Space:  O(n + |E|)

        The “threshold” constraint is always satisfiable (threshold ≥ 1)
        because the constructed arborescence gives every node
        out-degree ≤ 1 ≤ threshold.
        """
        # threshold is guaranteed to be at least 1, but keep the guard for safety
        if threshold < 1:
            return -1

        # Build reversed adjacency list
        adj_rev = [[] for _ in range(n)]
        for u, v, w in edges:
            adj_rev[v].append((u, w))     # reversed edge: v -> u with weight w

        INF = 10 ** 18
        dist = [INF] * n
        dist[0] = 0
        pq = [(0, 0)]                     # (current bottleneck, node)

        while pq:
            cur, node = heapq.heappop(pq)
            if cur != dist[node]:
                continue
            for nxt, w in adj_rev[node]:
                nxt_cost = max(cur, w)
                if nxt_cost < dist[nxt]:
                    dist[nxt] = nxt_cost
                    heapq.heappush(pq, (nxt_cost, nxt))

        # If some node cannot reach 0 in the original graph
        if any(d == INF for d in dist):
            return -1

        return max(dist)