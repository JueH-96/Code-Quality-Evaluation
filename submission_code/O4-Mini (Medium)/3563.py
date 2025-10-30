from collections import deque
from typing import List

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        # Build a flow network for max-weight bipartite matching between rows and values
        # We'll use min-cost max-flow with negative costs to pick largest weights.
        n = len(grid)
        # Collect distinct values
        vals = set()
        for row in grid:
            vals.update(row)
        vals = list(vals)
        m = len(vals)
        val_index = {v: i for i, v in enumerate(vals)}

        # Graph setup
        # nodes: 0 = source, 1..n = rows, n+1..n+m = values, n+m+1 = sink
        N = n + m + 2
        s = 0
        t = n + m + 1

        class Edge:
            __slots__ = ('to','cap','cost','rev')
            def __init__(self, to, cap, cost, rev):
                self.to = to
                self.cap = cap
                self.cost = cost
                self.rev = rev

        graph = [[] for _ in range(N)]

        def add_edge(u, v, cap, cost):
            # add forward edge u->v and backward edge v->u
            graph[u].append(Edge(v, cap, cost, len(graph[v])))
            graph[v].append(Edge(u, 0, -cost, len(graph[u]) - 1))

        # source -> each row
        for i in range(n):
            add_edge(s, 1 + i, 1, 0)
        # each value -> sink
        for j in range(m):
            add_edge(1 + n + j, t, 1, 0)
        # row -> value edges with cost = -value (to maximize sum)
        for i in range(n):
            seen = set()
            for v in grid[i]:
                if v in seen:  # skip duplicates in same row
                    continue
                seen.add(v)
                j = val_index[v]
                add_edge(1 + i, 1 + n + j, 1, -v)

        # Min-Cost Max-Flow via SPFA
        INF = 10**9
        total_cost = 0
        total_flow = 0

        while True:
            dist = [INF] * N
            in_queue = [False] * N
            pv = [-1] * N   # previous vertex
            pe = [-1] * N   # previous edge index
            dq = deque([s])
            dist[s] = 0
            in_queue[s] = True
            # SPFA to find shortest augmenting path
            while dq:
                u = dq.popleft()
                in_queue[u] = False
                for idx, e in enumerate(graph[u]):
                    if e.cap > 0 and dist[e.to] > dist[u] + e.cost:
                        dist[e.to] = dist[u] + e.cost
                        pv[e.to] = u
                        pe[e.to] = idx
                        if not in_queue[e.to]:
                            in_queue[e.to] = True
                            dq.append(e.to)
            if dist[t] == INF:
                break
            # augment 1 unit
            v = t
            while v != s:
                u = pv[v]
                idx = pe[v]
                e = graph[u][idx]
                e.cap -= 1
                graph[v][e.rev].cap += 1
                v = u
            total_flow += 1
            total_cost += dist[t]

        # total_cost is negative of the selected sum
        return -total_cost